//! Expression evaluation and shaped storage access over the checked model.
//! Every registry primitive has exactly one reference evaluation.
use super::exec::Frame;
use super::scalar;
use super::value::{Backing, Shaped, Value, S};
use super::{round_to, Interpreter, TensorData};
use crate::intrinsics::{accumulator_dtype, MathOp, PrimitiveId, ReduceOp};
use crate::repr::{self, Coefficient, PlaneEncoding};
use crate::sir::{CheckedExpr, CheckedExprKind, CheckedIndex, Literal};
use crate::syntax::ast::{AssignOp, BinaryOp};
use crate::types::{DType, Elem, ExtentExpr};

/// An elementwise operand: a broadcast scalar or row-major tensor data.
pub(super) enum Operand {
    Scalar(S),
    Tensor {
        dtype: DType,
        shape: Vec<usize>,
        data: Vec<f64>,
    },
}

pub(super) fn hint(ty: &crate::types::ValueType) -> Option<DType> {
    use crate::types::ValueType;
    match ty {
        ValueType::Scalar(d) => Some(*d),
        ValueType::Index { .. } => Some(DType::I32),
        ValueType::Tensor(s) => match s.elem {
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
                    Some(false) => Err("read of an uninitialized tensor element".into()),
                    None => Err("read outside tensor bounds".into()),
                }
            }
        }
    }

    pub(super) fn write_flat(&mut self, s: &Shaped, flat: usize, value: S) -> Result<(), String> {
        match &s.backing {
            Backing::Tensor(id) => match &mut self.tensors[*id] {
                TensorData::Packed { .. } => {
                    Err("packed representations are readable and decodable but not writable".into())
                }
                TensorData::Dense { dtype, data, .. } => {
                    let slot = data.get_mut(flat).ok_or("write outside tensor bounds")?;
                    *slot = scalar::cast(*dtype, value).1;
                    Ok(())
                }
            },
            Backing::Owned(rc) => {
                let mut dense = rc.borrow_mut();
                if flat >= dense.data.len() {
                    return Err("write outside tensor bounds".into());
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
            Value::Tensor(s) => Ok(Operand::Tensor {
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
            if let Operand::Tensor { shape: s, .. } = o {
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
                    Operand::Tensor { dtype, data, .. } => (*dtype, data[i]),
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
        Ok(Value::Tensor(Shaped::owned(dtype, shape.clone(), data)))
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
        if let Operand::Tensor { shape, .. } = &source {
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
                Operand::Tensor { dtype, data, .. } => (*dtype, data[i]),
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

    /// Value semantics at a binding or return: an owned tensor that anything
    /// else can still reach is copied. Views stay borrows; packed snapshots are
    /// immutable descriptors.
    pub(super) fn snapshot(&self, v: Value) -> Result<Value, String> {
        Ok(match v {
            Value::Tensor(s) if matches!(s.backing, Backing::Owned(_)) && !s.exclusive() => {
                Value::Tensor(Shaped::owned(
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

    pub(super) fn extent(&self, e: &ExtentExpr, f: &Frame<'a>) -> Result<usize, String> {
        let n = match e {
            ExtentExpr::Static(n) => *n as i64,
            ExtentExpr::Sym(s) => f.sym(s)?,
            ExtentExpr::Runtime(id) => {
                return Err(format!("runtime extent #{} is not bound here", id.0))
            }
        };
        usize::try_from(n).map_err(|_| format!("negative extent {n}"))
    }

    /// Apply checked indices to a selection. Nothing is clamped: selections
    /// outside the axis are errors.
    pub(super) fn select(
        &mut self,
        b: Shaped,
        indices: &[CheckedIndex],
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
            let point = |v: i64| -> Result<usize, String> {
                usize::try_from(v)
                    .ok()
                    .filter(|p| *p < extent)
                    .ok_or_else(|| format!("index {v} outside axis {axis} of extent {extent}"))
            };
            let (lo, hi) = match index {
                CheckedIndex::Point(p) => {
                    out.offset += point(self.int(p, f)?)? * stride;
                    continue;
                }
                CheckedIndex::Range { start, end } => {
                    let lo = match start {
                        Some(x) => self.int(x, f)?,
                        None => 0,
                    };
                    let hi = match end {
                        Some(x) => self.int(x, f)?,
                        None => extent as i64,
                    };
                    (lo, hi)
                }
            };
            if lo < 0 || hi < lo || hi > extent as i64 {
                return Err(format!(
                    "selection {}..{} outside axis {axis} of extent {extent}",
                    lo, hi
                ));
            }
            out.offset += lo as usize * stride;
            out.shape.push((hi - lo) as usize);
            out.strides.push(stride);
        }
        Ok(out)
    }

    /// The storage an expression designates, without reading it.
    pub(super) fn place(&mut self, e: &CheckedExpr, f: &mut Frame<'a>) -> Result<Shaped, String> {
        match &e.kind {
            CheckedExprKind::Primitive {
                id: PrimitiveId::SliceView { .. },
                operands,
            } => {
                let b = self.place(&operands[0], f)?;
                let indices = slots_of(e);
                self.select(b, &indices, f)
            }
            CheckedExprKind::Primitive {
                id: PrimitiveId::Transpose,
                operands,
            } => self.place(&operands[0], f)?.transposed(),
            CheckedExprKind::Primitive {
                id: PrimitiveId::Reshape,
                operands,
            } => {
                let base = self.place(&operands[0], f)?;
                let target: Vec<i64> = operands[1..]
                    .iter()
                    .map(|d| self.int(d, f).map(|n| n as i64))
                    .collect::<Result<_, _>>()?;
                let signed = |v: &[usize]| {
                    v.iter()
                        .map(|n| {
                            i64::try_from(*n).map_err(|_| "reshape extent overflow".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()
                };
                let strides = crate::layout::reshape_strides(
                    &signed(&base.shape)?,
                    &signed(&base.strides)?,
                    &target,
                )?;
                Ok(Shaped {
                    backing: base.backing.clone(),
                    shape: target.into_iter().map(|n| n as usize).collect(),
                    strides: strides.into_iter().map(|n| n as usize).collect(),
                    offset: base.offset,
                })
            }
            _ => match self.expr(e, f)? {
                Value::Tensor(s) => Ok(s),
                other => Err(format!("{} is not storage", other.kind())),
            },
        }
    }

    pub(super) fn scalar(&mut self, e: &CheckedExpr, f: &mut Frame<'a>) -> Result<S, String> {
        match self.expr(e, f)? {
            Value::Scalar(d, x) => Ok((d, x)),
            other => Err(format!("expected a scalar, found {}", other.kind())),
        }
    }

    pub(super) fn int(&mut self, e: &CheckedExpr, f: &mut Frame<'a>) -> Result<i64, String> {
        match self.scalar(e, f)? {
            (d, x) if d.is_int() => Ok(x as i64),
            (d, _) => Err(format!("expected an integer, found {}", d.name())),
        }
    }

    // ----- expressions -----------------------------------------------------------------

    pub(super) fn expr(&mut self, e: &CheckedExpr, f: &mut Frame<'a>) -> Result<Value, String> {
        match &e.kind {
            CheckedExprKind::Literal(literal) => Ok(match literal {
                Literal::Int(v) => {
                    let d = hint(&e.ty).unwrap_or(DType::I32);
                    Value::Scalar(
                        d,
                        if d.is_float() {
                            round_to(d, *v as f64)
                        } else {
                            *v as f64
                        },
                    )
                }
                Literal::Float(v) => {
                    let d = hint(&e.ty).filter(|d| d.is_float()).unwrap_or(DType::F32);
                    Value::Scalar(d, round_to(d, *v))
                }
                Literal::Bool(b) => Value::Scalar(DType::Bool, u8::from(*b) as f64),
                Literal::ShapeParam(name) => f
                    .shapes
                    .get(name)
                    .map(|v| Value::int(*v))
                    .ok_or_else(|| format!("shape parameter {name} is unbound"))?,
            }),
            CheckedExprKind::Local(id) => {
                f.locals.get(*id).and_then(|v| v.clone()).ok_or_else(|| {
                    format!(
                        "`{}` is read before it has a value",
                        f.body.locals[*id].name
                    )
                })
            }
            CheckedExprKind::Primitive { id, operands } => self.primitive(id, operands, e, f),
            CheckedExprKind::Capability { id, args } => self.capability(id, args, e, f),
            CheckedExprKind::Call { call, args } => self.call(call, args, f),
        }
    }

    /// One reference evaluation per registry primitive, exhaustive.
    fn primitive(
        &mut self,
        id: &PrimitiveId,
        operands: &[CheckedExpr],
        e: &CheckedExpr,
        f: &mut Frame<'a>,
    ) -> Result<Value, String> {
        match id {
            PrimitiveId::TuplePack => Ok(Value::Tuple(
                operands
                    .iter()
                    .map(|i| self.expr(i, f))
                    .collect::<Result<_, _>>()?,
            )),
            PrimitiveId::TupleGet(index) => {
                let index = *index;
                match self.expr(&operands[0], f)? {
                    Value::Tuple(mut items) if index < items.len() => Ok(items.swap_remove(index)),
                    other => Err(format!("component {index} of {}", other.kind())),
                }
            }
            PrimitiveId::RangeMake => {
                let lo = self.int(&operands[0], f)?;
                let hi = self.int(&operands[1], f)?;
                Ok(Value::Range(lo, hi))
            }
            PrimitiveId::RangeStart => match self.expr(&operands[0], f)? {
                Value::Range(lo, _) => Ok(Value::int(lo)),
                other => Err(format!("range start of {}", other.kind())),
            },
            PrimitiveId::RangeEnd => match self.expr(&operands[0], f)? {
                Value::Range(_, hi) => Ok(Value::int(hi)),
                other => Err(format!("range end of {}", other.kind())),
            },
            PrimitiveId::Unary(op) => {
                let v = self.expr(&operands[0], f)?;
                let operand = self.operand(&v)?;
                self.elementwise(&[operand], hint(&e.ty).unwrap_or(DType::F32), &mut |a| {
                    scalar::unary(*op, a[0])
                })
            }
            PrimitiveId::Binary(op) => {
                let l = self.expr(&operands[0], f)?;
                // Scalar logic short-circuits, so a guard protects its right operand.
                if let (BinaryOp::And | BinaryOp::Or, Value::Scalar(DType::Bool, x)) = (op, &l) {
                    if (*x != 0.0) == (*op == BinaryOp::Or) {
                        return Ok(l);
                    }
                }
                let r = self.expr(&operands[1], f)?;
                let h = hint(&e.ty);
                let operands = [self.operand(&l)?, self.operand(&r)?];
                self.elementwise(&operands, h.unwrap_or(DType::F32), &mut |a| {
                    scalar::binary(*op, a[0], a[1], h)
                })
            }
            PrimitiveId::Cast(dtype) => {
                let v = self.expr(&operands[0], f)?;
                let operand = self.operand(&v)?;
                self.elementwise(&[operand], *dtype, &mut |a| Ok(scalar::cast(*dtype, a[0])))
            }
            PrimitiveId::Math(op) => {
                let mut collected = Vec::with_capacity(operands.len());
                for a in operands {
                    let v = self.expr(a, f)?;
                    collected.push(self.operand(&v)?);
                }
                self.elementwise(&collected, hint(&e.ty).unwrap_or(DType::F32), &mut |a| {
                    scalar::math(*op, a)
                })
            }
            PrimitiveId::Select => {
                let c = self.expr(&operands[0], f)?;
                let t = self.expr(&operands[1], f)?;
                let n = self.expr(&operands[2], f)?;
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
            PrimitiveId::TensorAlloc { .. } => {
                let shape = operands
                    .iter()
                    .map(|d| self.extent_of_operand(d, f))
                    .collect::<Result<Vec<_>, _>>()?;
                let dtype = match &e.ty {
                    crate::types::ValueType::Tensor(s) => f.elem(&s.elem),
                    _ => return Err("owned allocation without a tensor type".into()),
                };
                match dtype {
                    Elem::Dtype(d) => Ok(Value::Tensor(Shaped::uninit(d, shape))),
                    other => Err(format!(
                        "local tensor allocation requires a dense dtype, found {other}"
                    )),
                }
            }
            PrimitiveId::Fill { value, dtype } => {
                let like = self.place(&operands[0], f)?;
                let n = like.count();
                Ok(Value::Tensor(Shaped::owned(
                    *dtype,
                    like.shape,
                    vec![round_to(*dtype, *value); n],
                )))
            }
            PrimitiveId::Materialize | PrimitiveId::Clone | PrimitiveId::Load => {
                let s = self.place(&operands[0], f)?;
                if matches!(self.elem_of(&s), Elem::Repr(_)) {
                    // A packed snapshot is an immutable descriptor over the same plane bytes.
                    return Ok(Value::Tensor(s));
                }
                Ok(Value::Tensor(Shaped::owned(
                    self.dtype_of(&s),
                    s.shape.clone(),
                    self.gather(&s)?,
                )))
            }
            PrimitiveId::Decode => {
                let s = self.place(&operands[0], f)?;
                let data = self
                    .gather(&s)?
                    .into_iter()
                    .map(|x| round_to(DType::F32, x))
                    .collect();
                Ok(Value::Tensor(Shaped::owned(DType::F32, s.shape, data)))
            }
            PrimitiveId::PackedRead(field) => {
                let s = self.place(&operands[0], f)?;
                self.packed_read(&s, field.name())
            }
            PrimitiveId::Transpose => Ok(match self.expr(&operands[0], f)? {
                Value::Tensor(s) => Value::Tensor(s.transposed()?),
                other => return Err(format!("transpose of {}", other.kind())),
            }),
            PrimitiveId::Reshape => self.place(e, f).map(Value::Tensor),
            PrimitiveId::SliceView { .. } => self.place(e, f).map(Value::Tensor),
            PrimitiveId::ElementRead { .. } => {
                let s = self.place(&operands[0], f)?;
                let indices = point_indices(operands);
                let selected = self.select(s, &indices, f)?;
                if !selected.shape.is_empty() {
                    return Err("a point read selects one element".into());
                }
                Ok(Value::Scalar(
                    self.dtype_of(&selected),
                    self.read_flat(&selected, selected.offset)?,
                ))
            }
            PrimitiveId::ElementWrite { .. } | PrimitiveId::CopyInto => {
                Err("write primitives are carried by checked assignments, not expressions".into())
            }
            PrimitiveId::Extent { axis } | PrimitiveId::ValidExtent { axis } => {
                let s = self.place(&operands[0], f)?;
                s.shape
                    .get(*axis)
                    .map(|n| Value::int(*n as i64))
                    .ok_or_else(|| {
                        format!("extent of axis {axis} of a rank-{} value", s.shape.len())
                    })
            }
            PrimitiveId::Atomic { op, .. } => {
                // The place is the operand prefix `[base, indices…]`, the value last.
                let value = self.scalar(operands.last().expect("atomic has a value"), f)?;
                let indices: Vec<CheckedIndex> = operands[1..operands.len() - 1]
                    .iter()
                    .map(|p| CheckedIndex::Point(p.clone()))
                    .collect();
                let b = self.expr(&operands[0], f)?;
                let Value::Tensor(shaped) = b else {
                    return Err("the `atomic` place does not name tensor storage".into());
                };
                let dst = self.select(shaped, &indices, f)?;
                if !dst.shape.is_empty() {
                    return Err("atomic update of a place that is not one element".into());
                }
                let d = self.dtype_of(&dst);
                let current = (d, self.read_flat(&dst, dst.offset)?);
                let updated = match op {
                    // Registry `RoundsOnce`: the sum rounds once at the element dtype.
                    crate::intrinsics::AtomicOp::Add => {
                        scalar::binary(BinaryOp::Add, current, value, Some(d))?
                    }
                    // Exact selection of one operand; a NaN operand is ignored,
                    // as in the reference `max`/`min` reductions.
                    crate::intrinsics::AtomicOp::Max => (d, current.1.max(value.1)),
                    crate::intrinsics::AtomicOp::Min => (d, current.1.min(value.1)),
                };
                self.write_flat(&dst, dst.offset, updated)?;
                Ok(Value::Void)
            }
            PrimitiveId::Reduce { op, axis, .. } => {
                let s = self.place(&operands[0], f)?;
                self.reduce(&s, *axis, *op)
            }
        }
    }

    fn extent_of_operand(&mut self, d: &CheckedExpr, f: &mut Frame<'a>) -> Result<usize, String> {
        match &d.ty {
            crate::types::ValueType::Index { bound } => self.extent(bound, f),
            _ => {
                let n = self.int(d, f)?;
                usize::try_from(n).map_err(|_| format!("negative extent {n}"))
            }
        }
    }

    /// Reduction along one axis in ascending coordinate order, with the
    /// registry's accumulator, identity and tie semantics. Integer sums wrap
    /// at the operand dtype each step; `argmax` requires a nonempty axis and
    /// keeps the smaller coordinate on ties.
    fn reduce(&self, s: &Shaped, axis: usize, op: ReduceOp) -> Result<Value, String> {
        if axis >= s.shape.len() {
            return Err(format!(
                "reduce along axis {axis} of a rank-{} value",
                s.shape.len()
            ));
        }
        let input = self.dtype_of(s);
        if matches!(self.elem_of(s), Elem::Repr(_)) {
            return Err("reduce requires a dense tile; decode packed values first".into());
        }
        let accumulator = accumulator_dtype(op, input);
        let data = self.gather(s)?;
        let extent = s.shape[axis];
        if extent == 0 && op == ReduceOp::Argmax {
            return Err("argmax requires a nonempty axis".into());
        }
        let outer: usize = s.shape[..axis].iter().product();
        let inner: usize = s.shape[axis + 1..].iter().product();
        let mut out = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = match (op, accumulator) {
                    (ReduceOp::Sum, _) => 0.0,
                    (ReduceOp::Min, DType::Bool) => 1.0,
                    (_, DType::Bool) => 0.0,
                    (ReduceOp::Min, DType::I32) => f64::from(i32::MAX),
                    (_, DType::I32) => f64::from(i32::MIN),
                    (ReduceOp::Min, DType::U32) => f64::from(u32::MAX),
                    (_, DType::U32) => 0.0,
                    (ReduceOp::Min, _) => f64::INFINITY,
                    _ => f64::NEG_INFINITY,
                };
                let mut arg = 0usize;
                for k in 0..extent {
                    let x = data[(o * extent + k) * inner + i];
                    match op {
                        ReduceOp::Sum => acc = round_to(accumulator, acc + x),
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
        let dtype = accumulator_dtype(op, input);
        Ok(if shape.is_empty() {
            Value::Scalar(dtype, out[0])
        } else {
            Value::Tensor(Shaped::owned(dtype, shape, out))
        })
    }

    // ----- capability intrinsics ---------------------------------------------------------

    /// Capability intrinsics under a single participant: the reference model
    /// has one participant per group, so an exchange returns its own value and
    /// a subgroup reduction of one value is that value. Logical matrix
    /// intrinsics evaluate their defined reference order.
    fn capability(
        &mut self,
        id: &crate::intrinsics::IntrinsicId,
        args: &[CheckedExpr],
        e: &CheckedExpr,
        f: &mut Frame<'a>,
    ) -> Result<Value, String> {
        match id.name.as_str() {
            "lane_index" => Ok(Value::int(0)),
            "shuffle" => {
                let v = self.expr(&args[0], f)?;
                match self.int(&args[1], f)? {
                    0 => Ok(v),
                    other => Err(format!("participant {other} outside a group of one")),
                }
            }
            "simd_sum" | "simd_max" | "simd_min" => self.expr(&args[0], f),
            "matmul" | "matmul_add" => self.matmul(id, args, e, f),
            other => Err(format!("`{other}` has no reference semantics")),
        }
    }

    /// The reference order: each output accumulates an fma chain over an
    /// ascending inner axis, rounding at the accumulation dtype.
    fn matmul(
        &mut self,
        id: &crate::intrinsics::IntrinsicId,
        args: &[CheckedExpr],
        e: &CheckedExpr,
        f: &mut Frame<'a>,
    ) -> Result<Value, String> {
        let add = id.name == "matmul_add";
        let expected = if add { 3 } else { 2 };
        if args.len() != expected {
            return Err(format!("`{id}` takes {expected} operands"));
        }
        let left = self.place(&args[0], f)?;
        let right = self.place(&args[1], f)?;
        if left.shape.len() != 2 || right.shape.len() != 2 {
            return Err(format!("`{id}` operands must be rank two"));
        }
        let (rows, inner, columns) = (left.shape[0], left.shape[1], right.shape[1]);
        if right.shape[0] != inner {
            return Err(format!(
                "`{id}` inner extents differ: {inner} and {}",
                right.shape[0]
            ));
        }
        let crate::types::ValueType::Tensor(result) = &e.ty else {
            return Err(format!("`{id}` result is not an owned tensor value"));
        };
        let Elem::Dtype(dtype) = f.elem(&result.elem) else {
            return Err(format!("`{id}` result must have a dense element type"));
        };
        let left_dtype = self.dtype_of(&left);
        let right_dtype = self.dtype_of(&right);
        let left_values = self.gather(&left)?;
        let right_values = self.gather(&right)?;
        let accumulator = if add {
            let value = self.place(&args[2], f)?;
            if value.shape != [rows, columns] {
                return Err(format!(
                    "`{id}` accumulator shape {:?} differs from [{rows}, {columns}]",
                    value.shape
                ));
            }
            Some((self.dtype_of(&value), self.gather(&value)?))
        } else {
            None
        };
        let mut output = Vec::with_capacity(rows.saturating_mul(columns));
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = accumulator.as_ref().map_or(
                    scalar::cast(dtype, (DType::I32, 0.0)),
                    |(source_dtype, values)| {
                        scalar::cast(dtype, (*source_dtype, values[row * columns + column]))
                    },
                );
                for k in 0..inner {
                    let l = (left_dtype, left_values[row * inner + k]);
                    let r = (right_dtype, right_values[k * columns + column]);
                    sum = if dtype.is_int() {
                        let product = scalar::binary(
                            BinaryOp::Mul,
                            scalar::cast(dtype, l),
                            scalar::cast(dtype, r),
                            Some(dtype),
                        )?;
                        scalar::binary(BinaryOp::Add, sum, product, Some(dtype))?
                    } else {
                        scalar::cast(dtype, scalar::math(MathOp::Fma, &[l, r, sum])?)
                    };
                }
                output.push(sum.1);
            }
        }
        Ok(Value::Tensor(Shaped::owned(
            dtype,
            vec![rows, columns],
            output,
        )))
    }

    // ----- packed planes -----------------------------------------------------------------

    /// `t.words`, `t.scale`, `t.bias` and the other physical planes of a packed
    /// value: a dense tile whose packet axis is replaced by the plane's extent
    /// over that axis. Planes are readable and decodable, never writable.
    fn packed_read(&self, s: &Shaped, name: &str) -> Result<Value, String> {
        let Backing::Tensor(id) = &s.backing else {
            return Err(format!("`.{name}` needs a packed value"));
        };
        let TensorData::Packed {
            repr: rep, planes, ..
        } = &self.tensors[*id]
        else {
            return Err(format!("`.{name}` needs a packed value"));
        };
        let axis = s.shape.len().saturating_sub(1);
        if axis >= s.shape.len() || s.strides[axis] != 1 {
            return Err(format!("`.{name}` needs the packet axis in storage order"));
        }
        let length = s.shape[axis];
        let logical = name == "scale" || name == "bias";
        let plane = if logical {
            None
        } else {
            Some(
                rep.plane(name)
                    .ok_or_else(|| format!("`{}` has no physical plane `{name}`", rep.name))?,
            )
        };
        let group = plane.as_ref().map_or(rep.group, |p| p.group) as usize;
        let (count, dtype) = match &plane {
            None => (length.div_ceil(group), rep.coefficient_dtype()),
            Some(p) => (
                p.storage_elements(length as u64)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("plane extent overflow")?,
                p.dtype(),
            ),
        };
        let mut rows = s.clone();
        rows.shape[axis] = 1;
        let mut data = Vec::with_capacity(rows.count() * count);
        // Rows are visited in row-major order of the result, whose packet axis is `axis`.
        let outer: usize = s.shape[..axis].iter().product();
        let inner: usize = s.shape[axis + 1..].iter().product();
        let starts = rows.flats();
        for o in 0..outer {
            for e in 0..count {
                for i in 0..inner {
                    let first = starts[o * inner + i];
                    if first % group != 0 {
                        return Err(format!(
                            "`.{name}` of a selection that does not start on a group of {group}"
                        ));
                    }
                    data.push(match &plane {
                        None => coefficient_value(rep, planes, name == "bias", first + e * group)?,
                        Some(p) => {
                            let entry = first / group * p.fields as usize;
                            match &p.encoding {
                                PlaneEncoding::Dense(_) => {
                                    plane_value(rep, planes, p, entry + e)?
                                }
                                PlaneEncoding::Packed { bits, .. } => {
                                    let bit = entry * *bits as usize;
                                    if bit % 32 != 0 {
                                        return Err(format!("`.{name}` of a selection that does not start on a storage word"));
                                    }
                                    let bytes = rep.plane_index(p.name).and_then(|k| planes.get(k)).ok_or("missing plane storage")?;
                                    let at = (bit / 32 + e) * 4;
                                    let word: [u8; 4] =
                                        std::array::from_fn(|k| bytes.get(at + k).copied().unwrap_or(0));
                                    f64::from(u32::from_le_bytes(word))
                                }
                            }
                        }
                    });
                }
            }
        }
        let mut shape = s.shape.clone();
        shape[axis] = count;
        Ok(Value::Tensor(Shaped::owned(dtype, shape, data)))
    }

    #[cfg(test)]
    pub(crate) fn gather_for_test(&self, s: &Shaped) -> Vec<f64> {
        self.gather(s).unwrap()
    }
}

/// The checked indices of a point read: the operand exprs after the base.
fn point_indices(operands: &[CheckedExpr]) -> Vec<CheckedIndex> {
    operands[1..]
        .iter()
        .map(|p| CheckedIndex::Point(p.clone()))
        .collect()
}

/// The index slots of a view-selection expression, reconstructed from its
/// operand structure: `[base, start0, end0, …]` in slot order.
fn slots_of(e: &CheckedExpr) -> Vec<CheckedIndex> {
    let CheckedExprKind::Primitive {
        id: PrimitiveId::SliceView { indices },
        operands,
    } = &e.kind
    else {
        return Vec::new();
    };
    let mut present = operands[1..].iter();
    indices
        .iter()
        .map(|slot| match slot {
            crate::intrinsics::IndexSlot::Point => CheckedIndex::Point(
                present
                    .next()
                    .cloned()
                    .expect("operand count matches slots"),
            ),
            crate::intrinsics::IndexSlot::Range { start, end } => CheckedIndex::Range {
                start: if *start {
                    present.next().cloned()
                } else {
                    None
                },
                end: if *end { present.next().cloned() } else { None },
            },
        })
        .collect()
}

/// One decoded entry of a physical plane.
fn plane_value(
    rep: &repr::Repr,
    planes: &[Vec<u8>],
    plane: &repr::Plane,
    entry: usize,
) -> Result<f64, String> {
    let bytes = rep
        .plane_index(plane.name)
        .and_then(|i| planes.get(i))
        .ok_or_else(|| format!("`{}` has no plane `{}`", rep.name, plane.name))?;
    let outside = || {
        format!(
            "plane `{}` entry {entry} is outside its storage",
            plane.name
        )
    };
    match &plane.encoding {
        PlaneEncoding::Packed {
            bits,
            interpretation,
        } => {
            if (entry + 1) * *bits as usize > bytes.len() * 8 {
                return Err(outside());
            }
            Ok(f64::from(
                interpretation.decode(repr::read_packed(bytes, entry, *bits), *bits),
            ))
        }
        PlaneEncoding::Dense(dtype) => {
            let width = dtype.bytes() as usize;
            let raw = bytes
                .get(entry * width..(entry + 1) * width)
                .ok_or_else(outside)?;
            Ok(f64::from(match (dtype, raw) {
                (DType::F32, [a, b, c, d]) => f32::from_le_bytes([*a, *b, *c, *d]),
                (DType::F16, [a, b]) => crate::numeric::f16_to_f32(u16::from_le_bytes([*a, *b])),
                (DType::BF16, [a, b]) => {
                    f32::from_bits(u32::from(u16::from_le_bytes([*a, *b])) << 16)
                }
                _ => {
                    return Err(format!(
                        "plane `{}` is not a floating coefficient plane",
                        plane.name
                    ))
                }
            }))
        }
    }
}

/// The logical scale or bias applying to the value at flat position `flat`.
fn coefficient_value(
    rep: &repr::Repr,
    planes: &[Vec<u8>],
    bias: bool,
    flat: usize,
) -> Result<f64, String> {
    match rep.coefficient(bias) {
        None => Err(format!("`{}` has no bias", rep.name)),
        Some(Coefficient::Direct { plane }) => {
            plane_value(rep, planes, &plane, flat / plane.group as usize)
        }
        Some(Coefficient::Product {
            factor,
            coefficients,
            field,
            sign,
        }) => {
            let code = plane_value(
                rep,
                planes,
                &coefficients,
                flat / coefficients.group as usize * coefficients.fields as usize + field as usize,
            )?;
            let factor = plane_value(rep, planes, &factor, flat / factor.group as usize)?;
            Ok(f64::from((factor as f32 * code as f32) * sign as f32))
        }
    }
}
