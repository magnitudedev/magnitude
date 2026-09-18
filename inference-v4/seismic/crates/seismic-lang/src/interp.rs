//! Reference interpreter: direct evaluation of portable IR. This is the
//! semantics of the language and the oracle for every lowering.
//!
//! Tensors are host arrays. Dense elements are stored as f64 but every write
//! rounds to the tensor's dtype, and every arithmetic operation is performed
//! at the dtype the checker assigned, so results carry the same rounding a
//! device would apply. Packed tensors hold their words and coefficients and
//! decode on read.

use crate::numeric::{bf16_round, f16_round, f16_bits, f16_to_f32};
use crate::ast::{AssignOp, BinaryOp, UnaryOp};
use crate::ir::*;
use crate::program::Program;
use crate::repr;
use crate::sym::{Atom, Sym};
use crate::types::{DType, Elem, Ty};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum TensorData {
    Dense { dtype: DType, shape: Vec<usize>, data: Vec<f64> },
    /// Physical byte planes in representation ABI order.
    Packed { repr: &'static repr::Repr, shape: Vec<usize>, planes: Vec<Vec<u8>> },
}

impl TensorData {
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorData::Dense { shape, .. } | TensorData::Packed { shape, .. } => shape,
        }
    }

    pub fn dense(dtype: DType, shape: Vec<usize>, data: Vec<f64>) -> TensorData {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        TensorData::Dense { dtype, shape, data }
    }

    /// Decoded value at a flat row-major position.
    pub fn get(&self, flat: usize) -> f64 {
        match self {
            TensorData::Dense { data, .. } => data[flat],
            TensorData::Packed { repr, planes, .. } => {
                let plane_value = |plane: &repr::Plane, entry: usize| -> f32 {
                    let bytes = &planes[repr.plane_index(plane.name).unwrap()];
                    match &plane.encoding {
                        repr::PlaneEncoding::Packed { bits, interpretation } => interpretation.decode(repr::read_packed(bytes, entry, *bits), *bits) as f32,
                        repr::PlaneEncoding::Dense(dtype) => {
                            let start = entry * dtype.bytes() as usize;
                            match dtype {
                                DType::F32 => f32::from_le_bytes(bytes[start..start+4].try_into().unwrap()),
                                DType::F16 => f16_to_f32(u16::from_le_bytes(bytes[start..start+2].try_into().unwrap())),
                                DType::BF16 => f32::from_bits(u32::from(u16::from_le_bytes(bytes[start..start+2].try_into().unwrap())) << 16),
                                _ => unreachable!("nonfloating coefficient"),
                            }
                        }
                    }
                };
                let coefficient = |bias| match repr.coefficient(bias) {
                    None => 0.0,
                    Some(repr::Coefficient::Direct { plane }) => plane_value(&plane, flat / plane.group as usize),
                    Some(repr::Coefficient::Product { factor, coefficients, field, sign }) => {
                        let code = plane_value(&coefficients, flat / coefficients.group as usize * coefficients.fields as usize + field as usize);
                        (plane_value(&factor, flat / factor.group as usize) * code) * sign as f32
                    }
                };
                let code = repr::read_packed(&planes[0], flat, repr.bits);
                (coefficient(false) as f64 * repr.decode_code(code) as f64 + coefficient(true) as f64) as f32 as f64
            }
        }
    }

    pub fn set(&mut self, flat: usize, value: f64) {
        match self {
            TensorData::Dense { dtype, data, .. } => data[flat] = round_to(*dtype, value),
            TensorData::Packed { .. } => panic!("cannot store into a packed tensor"),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            TensorData::Dense { dtype, shape, .. } => shape.iter().product::<usize>() * dtype.bytes() as usize,
            TensorData::Packed { planes, .. } => planes.iter().map(Vec::len).sum(),
        }
    }
}

/// Round an f64 to a dtype's representable value.
pub fn round_to(dtype: DType, v: f64) -> f64 {
    match dtype {
        DType::F32 => v as f32 as f64,
        DType::BF16 => bf16_round(v as f32) as f64,
        DType::F16 => f16_round(v as f32) as f64,
        DType::I32 => v as i32 as f64,
        DType::U32 => v as u32 as f64,
        DType::Bool => if v != 0.0 { 1.0 } else { 0.0 },
    }
}

#[derive(Clone, Debug)]
struct View {
    tensor: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl View {
    fn flat(&self, idx: &[usize]) -> usize {
        self.offset + idx.iter().zip(&self.strides).map(|(i, s)| i * s).sum::<usize>()
    }
}

#[derive(Clone, Debug)]
struct Tile {
    shape: Vec<usize>,
    dtype: DType,
    data: Vec<f64>,
}

impl Tile {
    fn flat(&self, idx: &[usize]) -> usize {
        let mut f = 0;
        for (i, d) in idx.iter().zip(&self.shape) {
            f = f * d + i;
        }
        f
    }
}

#[derive(Clone, Debug)]
enum Value {
    Scalar(f64),
    Int(i64),
    Bool(bool),
    View(View),
    Tile(Tile),
    Tuple(Vec<Value>),
    Void,
}

pub struct Interpreter<'a> {
    program: &'a Program,
    pub tensors: Vec<TensorData>,
    vars_stack: Vec<Vec<Var>>,
}

struct Frame {
    vars: Vec<Option<Value>>,
    index: HashMap<String, i64>,
    shapes: HashMap<String, i64>,
    elements: HashMap<String, Elem>,
}

impl<'a> Interpreter<'a> {
    pub fn new(program: &'a Program) -> Interpreter<'a> {
        Interpreter { program, tensors: Vec::new(), vars_stack: Vec::new() }
    }

    pub fn add_tensor(&mut self, t: TensorData) -> usize {
        self.tensors.push(t);
        self.tensors.len() - 1
    }

    /// Run a function. `args` are tensor ids for tensor parameters and scalars otherwise;
    /// `shapes` binds every shape parameter.
    pub fn run(&mut self, name: &str, args: &[Arg], shapes: &HashMap<String, i64>) -> Result<(), String> {
        let f = self.program.functions.iter().find(|f| f.name == name).ok_or_else(|| format!("no function `{name}`"))?.clone();
        let mut frame = Frame { vars: vec![None; f.vars.len()], index: HashMap::new(), shapes: shapes.clone(), elements: HashMap::new() };
        if args.len() != f.params.len() {
            return Err(format!("`{name}` takes {} arguments", f.params.len()));
        }
        for (i, (arg, (parameter_name, ty))) in args.iter().zip(&f.params).enumerate() {
            let value = match (arg, ty) {
                (Arg::Tensor(id), Ty::Tensor(s)) => {
                    let shape: Vec<usize> = s.shape.iter().map(|d| self.eval_sym(d, &frame) as usize).collect();
                    if self.tensors[*id].shape() != shape.as_slice() {
                        return Err(format!("argument {i} has shape {:?}, expected {:?}", self.tensors[*id].shape(), shape));
                    }
                    if let Elem::Param(p) = &s.elem {
                        let actual = match &self.tensors[*id] {
                            TensorData::Dense { dtype, .. } => Elem::Dtype(*dtype),
                            TensorData::Packed { repr, .. } => Elem::Repr(repr.name.into()),
                        };
                        if frame.elements.insert(p.clone(), actual.clone()).is_some_and(|previous| previous != actual) {
                            return Err(format!("inconsistent element parameter {p}"));
                        }
                    }
                    Value::View(full_view(*id, &shape))
                }
                (Arg::Scalar(v), Ty::Scalar(d)) => {
                    let mut parameter=crate::abi::ScalarParameter::plain(parameter_name,*d);
                    parameter.index_bound=f.index_params.iter().find(|(n,_)|n==parameter_name).map(|(_,bound)|bound.eval(&|n|shapes.get(n).copied()).and_then(|n|u64::try_from(n).ok()).ok_or_else(||format!("unresolved index bound for `{parameter_name}`"))).transpose()?;
                    crate::abi::ScalarLayout::words(&[parameter])?.encode(&[*v])?;
                    if *d==DType::Bool { Value::Bool(*v==1.0) } else if d.is_int() {
                        if let VarKind::Index(Atom::Param(p)) = &f.vars[i].kind {
                            frame.index.insert(p.clone(), *v as i64);
                        }
                        Value::Int(*v as i64)
                    } else {
                        Value::Scalar(round_to(*d, *v))
                    }
                }
                _ => return Err(format!("argument {i} does not match parameter type {ty}")),
            };
            frame.vars[i] = Some(value);
        }
        self.vars_stack.push(f.vars.clone());
        let result = self.block(&f.body, &mut frame);
        self.vars_stack.pop();
        result
    }

    fn eval_sym(&self, s: &Sym, frame: &Frame) -> i64 {
        s.eval(&|p| frame.index.get(p).copied().or_else(|| frame.shapes.get(p).copied())).unwrap_or_else(|| panic!("unbound symbol in `{s}`"))
    }

    fn block(&mut self, stmts: &[Stmt], frame: &mut Frame) -> Result<(), String> {
        for s in stmts {
            self.stmt(s, frame)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt, frame: &mut Frame) -> Result<(), String> {
        match &s.kind {
            StmtKind::Reduction(reduction) => {
                let original = frame.vars.len();
                let vars = self.vars_stack.last_mut().unwrap();
                let body = reduction.expand(reduction.tree.unwrap_or(crate::reduction::structured::Tree::Ordered), vars)?;
                frame.vars.resize(vars.len(), None);
                let result = self.block(&body, frame);
                frame.vars.truncate(original);
                self.vars_stack.last_mut().unwrap().truncate(original);
                result
            }
            StmtKind::Parallel { vars, extents, body } => {
                let ext: Vec<i64> = extents.iter().map(|e| self.eval_sym(e, frame)).collect();
                let names = self.index_names(vars);
                self.nested(vars, &ext, body, frame, &names)
            }
            StmtKind::Owned { vars, tile, body } => {
                let t = self.expr(tile, frame)?;
                let shape: Vec<i64> = match &t {
                    Value::Tile(t) => t.shape.iter().map(|d| *d as i64).collect(),
                    _ => return Err("owned() of a non-tile".into()),
                };
                let names = self.index_names(vars);
                self.nested(vars, &shape, body, frame, &names)
            }
            StmtKind::Range { var, lo, hi, body } => {
                let lo = self.eval_sym(lo, frame);
                let hi = self.eval_sym(hi, frame);
                let name = self.index_names(&[*var])[0].clone();
                for i in lo..hi {
                    frame.index.insert(name.clone(), i);
                    frame.vars[*var] = Some(Value::Int(i));
                    self.block(body, frame)?;
                }
                Ok(())
            }
            StmtKind::Lanes { .. } => Err("lanes cannot be interpreted; interpret the portable body".into()),
            StmtKind::LoadLoop { domain, offset, vars, views, axes, piece, body, .. } => {
                // The interpreter takes the whole axis as one piece; pieces are a lowering choice.
                let mut tiles = Vec::new();
                let extent = match self.expr(&domain.view,frame)? {Value::View(v)=>v.shape[domain.axis],Value::Tile(t)=>t.shape[domain.axis],other=>return Err(format!("iteration domain is not shaped: {other:?}"))};
                for (v,axis) in views.iter().zip(axes) {
                    let tile = match self.expr(v, frame)? {
                        Value::View(view) => self.materialize(&view),
                        Value::Tile(tile) => tile,
                        other => return Err(format!("streamed binding is not shaped: {other:?}")),
                    };
                    if extent != tile.shape[*axis] {return Err("streamed binding extent differs from logical domain".into());}
                    tiles.push(tile);
                }
                if extent == 0 { return Ok(()); }
                if let Some(id)=offset {let VarKind::Index(Atom::Param(name))=&self.vars_stack.last().ok_or("missing interpreter variable scope")?[*id].kind else{return Err("stream offset must be an index".into())};frame.index.insert(name.clone(),0);frame.vars[*id]=Some(Value::Int(0));}
                let Atom::Param(pname) = piece else { unreachable!() };
                frame.index.insert(pname.clone(), extent as i64);
                for (var, t) in vars.iter().zip(tiles) {
                    frame.vars[*var] = Some(Value::Tile(t));
                }
                self.block(body, frame)
            }
            StmtKind::If { cond, then, els } => {
                match self.expr(cond, frame)? {
                    Value::Bool(true) => self.block(then, frame),
                    Value::Bool(false) => self.block(els, frame),
                    other => Err(format!("condition evaluated to {other:?}")),
                }
            }
            StmtKind::Assign { target, op, value } => {
                let v = self.expr(value, frame)?;
                match &target.kind {
                    ExprKind::Var(id) => {
                        let new = match (op, frame.vars[*id].clone(), &v) {
                            (AssignOp::Assign, _, _) => v,
                            (op, Some(Value::Scalar(a)), Value::Scalar(b)) => {
                                let d = scalar_dtype(&target.ty);
                                Value::Scalar(round_to(d, arith(*op, a, *b)))
                            }
                            (op, Some(Value::Int(a)), Value::Int(b)) => Value::Int(integer_arith(*op, scalar_dtype(&target.ty), a, *b)),
                            (op, Some(Value::Tile(mut t)), Value::Tile(b)) => {
                                for (x, y) in t.data.iter_mut().zip(&b.data) {
                                    *x = typed_arith(*op, t.dtype, *x, *y);
                                }
                                Value::Tile(t)
                            }
                            (op, a, b) => return Err(format!("unsupported `{}` on {a:?} and {b:?}", op.text())),
                        };
                        if let (VarKind::Index(Atom::Param(p)), Value::Int(i)) = (&self.vars_stack.last().unwrap()[*id].kind, &new) {
                            frame.index.insert(p.clone(), *i);
                        }
                        frame.vars[*id] = Some(new);
                        Ok(())
                    }
                    ExprKind::Index { base, indices } => {
                        let ExprKind::Var(id) = base.kind else { return Err("element assignment to a non-variable".into()) };
                        let idx: Vec<usize> = indices
                            .iter()
                            .map(|i| match i {
                                Index::Point(e) => match self.expr(e, frame)? {
                                    Value::Int(v) => usize::try_from(v).map_err(|_| "negative point index".to_string()),
                                    _ => Err("point index is not an integer".into()),
                                },
                                _ => Err("slice in element assignment".to_string()),
                            })
                            .collect::<Result<_, _>>()?;
                        let scalar = match v {
                            Value::Scalar(x) => x,
                            Value::Int(x) => x as f64,
                            Value::Bool(b) => b as i64 as f64,
                            other => return Err(format!("assigning {other:?} to an element")),
                        };
                        let Some(Value::Tile(t)) = frame.vars[id].as_mut() else { return Err("element assignment to a non-tile".into()) };
                        if idx.len() != t.shape.len() || idx.iter().zip(&t.shape).any(|(i,n)| i >= n) { return Err("point index outside tile bounds".into()); }
                        let flat = t.flat(&idx);
                        let cur = t.data[flat];
                        t.data[flat] = round_to(t.dtype, match op {
                            AssignOp::Assign => scalar,
                            _ => typed_arith(*op, t.dtype, cur, scalar),
                        });
                        Ok(())
                    }
                    _ => Err("unsupported assignment target".into()),
                }
            }
            StmtKind::Expr(e) => {
                self.expr(e, frame)?;
                Ok(())
            }
        }
    }

    /// The symbol names of a loop's index variables. Only these are needed per statement, so
    /// the variable table itself is never copied.
    fn index_names(&self, vars: &[VarId]) -> Vec<String> {
        let table = self.vars_stack.last().expect("no frame");
        vars.iter()
            .map(|v| match &table[*v].kind {
                VarKind::Index(Atom::Param(p)) => p.clone(),
                _ => unreachable!(),
            })
            .collect()
    }

    fn nested(&mut self, vars: &[VarId], extents: &[i64], body: &[Stmt], frame: &mut Frame, names: &[String]) -> Result<(), String> {
        let names: Vec<String> = names.to_vec();
        let mut idx = vec![0i64; vars.len()];
        if extents.iter().any(|e| *e <= 0) {
            return Ok(());
        }
        loop {
            for (k, v) in vars.iter().enumerate() {
                frame.index.insert(names[k].clone(), idx[k]);
                frame.vars[*v] = Some(Value::Int(idx[k]));
            }
            self.block(body, frame)?;
            let mut k = vars.len();
            loop {
                if k == 0 {
                    return Ok(());
                }
                k -= 1;
                idx[k] += 1;
                if idx[k] < extents[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
    }

    fn materialize(&self, view: &View) -> Tile {
        let n: usize = view.shape.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut idx = vec![0usize; view.shape.len()];
        let dtype = match &self.tensors[view.tensor] {
            TensorData::Dense { dtype, .. } => *dtype,
            TensorData::Packed { .. } => DType::F32,
        };
        for _ in 0..n {
            data.push(self.tensors[view.tensor].get(view.flat(&idx)));
            let mut k = idx.len();
            while k > 0 {
                k -= 1;
                idx[k] += 1;
                if idx[k] < view.shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        Tile { shape: view.shape.clone(), dtype, data }
    }

    fn expr(&mut self, e: &Expr, frame: &mut Frame) -> Result<Value, String> {
        match &e.kind {
            ExprKind::Load { view, .. } => self.builtin(Builtin::Load, std::slice::from_ref(view), frame),
            ExprKind::Int(v) => Ok(if scalar_dtype(&e.ty).is_float() { Value::Scalar(*v as f64) } else { Value::Int(*v) }),
            ExprKind::ShapeParam(_) => Ok(Value::Int(self.eval_sym(e.sym.as_ref().unwrap(), frame))),
            ExprKind::Float(v) => Ok(Value::Scalar(round_to(scalar_dtype(&e.ty), *v))),
            ExprKind::Bool(b) => Ok(Value::Bool(*b)),
            ExprKind::Var(id) => {
                if let Some(s) = &e.sym {
                    if frame.vars[*id].is_none() || matches!(frame.vars[*id], Some(Value::Int(_))) {
                        return Ok(Value::Int(self.eval_sym(s, frame)));
                    }
                }
                frame.vars[*id].clone().ok_or_else(|| format!("variable {id} is unset"))
            }
            ExprKind::TileAlloc { shape, dtype } => {
                let shape: Vec<usize> = shape.iter().map(|d| self.eval_sym(d, frame) as usize).collect();
                let n = shape.iter().product();
                let resolved = crate::lower::subst_elem(dtype, &frame.elements);
                let Elem::Dtype(dtype) = resolved else {return Err("local tile allocation requires a resolved dense dtype".into())};
                Ok(Value::Tile(Tile { shape, dtype, data: vec![f64::NAN; n] }))
            }
            ExprKind::Index { base, indices } => {
                // Fast path: an element read of a tile or view variable, without cloning the tile.
                if let ExprKind::Var(id) = base.kind {
                    if matches!(e.ty, Ty::Scalar(_)) && indices.iter().all(|i| matches!(i, Index::Point(_))) {
                        let mut idx = Vec::with_capacity(indices.len());
                        for i in indices {
                            let Index::Point(p) = i else { unreachable!() };
                            let fast = p.sym.as_ref().and_then(|sym| sym.eval(&|q| frame.index.get(q).copied().or_else(|| frame.shapes.get(q).copied())));
                            match fast {
                                Some(v) => idx.push(usize::try_from(v).map_err(|_| "negative point index")?),
                                None => match self.expr(p, frame)? {
                                    Value::Int(v) => idx.push(usize::try_from(v).map_err(|_| "negative point index")?),
                                    other => return Err(format!("index {other:?}")),
                                },
                            }
                        }
                        match &frame.vars[id] {
                            Some(Value::Tile(t)) if idx.len() == t.shape.len() => {
                                if idx.iter().zip(&t.shape).any(|(i,n)| i >= n) { return Err("point index outside tile bounds".into()); }
                                let flat = t.flat(&idx);
                                let x = t.data[flat];
                                return Ok(scalar_value(t.dtype, x));
                            }
                            Some(Value::View(v)) if idx.len() == v.shape.len() => {
                                if idx.iter().zip(&v.shape).any(|(i,n)| i >= n) { return Err("point index outside view bounds".into()); }
                                let flat = v.flat(&idx);
                                let x = self.tensors[v.tensor].get(flat);
                                return Ok(scalar_value(scalar_dtype(&e.ty), x));
                            }
                            _ => {}
                        }
                    }
                }
                let b = self.expr(base, frame)?;
                let mut points = Vec::new();
                let mut slices: Vec<Option<(usize, usize)>> = Vec::new();
                for i in indices {
                    match i {
                        Index::Point(p) => {
                            let v = match self.expr(p, frame)? {
                                Value::Int(v) => v,
                                other => return Err(format!("index {other:?}")),
                            };
                            points.push(Some(usize::try_from(v).map_err(|_| "negative point index")?));
                            slices.push(None);
                        }
                        Index::Slice { start, end } => {
                            let s = match start {
                                Some(x) => match self.expr(x, frame)? {
                                    Value::Int(v) => v.max(0) as usize,
                                    other => return Err(format!("slice start {other:?}")),
                                },
                                None => 0,
                            };
                            let en = match end {
                                Some(x) => match self.expr(x, frame)? {
                                    Value::Int(v) => Some(v.max(0) as usize),
                                    other => return Err(format!("slice end {other:?}")),
                                },
                                None => None,
                            };
                            points.push(None);
                            slices.push(Some((s, en.unwrap_or(usize::MAX))));
                        }
                    }
                }
                let apply = |shape: &[usize], strides: &[usize], offset: usize| -> Result<(Vec<usize>, Vec<usize>, usize), String> {
                    let mut ns = Vec::new();
                    let mut nst = Vec::new();
                    let mut off = offset;
                    for (axis, extent) in shape.iter().enumerate() {
                        if axis < points.len() {
                            if let Some(p) = points[axis] {
                                if p >= *extent { return Err("point index outside view bounds".into()); }
                                off += p * strides[axis];
                            } else {
                                let (s, en) = slices[axis].unwrap();
                                let en = en.min(*extent);
                                let s = s.min(en);
                                off += s * strides[axis];
                                ns.push(en - s);
                                nst.push(strides[axis]);
                            }
                        } else {
                            ns.push(*extent);
                            nst.push(strides[axis]);
                        }
                    }
                    Ok((ns, nst, off))
                };
                match b {
                    Value::View(v) => {
                        let (shape, strides, offset) = apply(&v.shape, &v.strides, v.offset)?;
                        if shape.is_empty() && matches!(e.ty, Ty::Scalar(_)) {
                            let x = self.tensors[v.tensor].get(offset);
                            Ok(scalar_value(scalar_dtype(&e.ty), x))
                        } else {
                            Ok(Value::View(View { tensor: v.tensor, shape, strides, offset }))
                        }
                    }
                    Value::Tile(t) => {
                        let strides = row_major(&t.shape);
                        let (shape, strides, offset) = apply(&t.shape, &strides, 0)?;
                        if shape.is_empty() && matches!(e.ty, Ty::Scalar(_)) {
                            return Ok(scalar_value(t.dtype, t.data[offset]));
                        }
                        Ok(Value::Tile(gather(&t, &shape, &strides, offset)))
                    }
                    Value::Tuple(items) => {
                        let mut out = Vec::new();
                        for item in items {
                            let sub = Expr { kind: ExprKind::Tuple(vec![]), ty: Ty::Void, sym: None, span: e.span };
                            let _ = sub;
                            out.push(self.index_value(item, &points, &slices)?);
                        }
                        Ok(Value::Tuple(out))
                    }
                    other => Err(format!("cannot index {other:?}")),
                }
            }
            ExprKind::Transpose(inner) => match self.expr(inner, frame)? {
                Value::View(v) => Ok(Value::View(View { tensor: v.tensor, shape: vec![v.shape[1], v.shape[0]], strides: vec![v.strides[1], v.strides[0]], offset: v.offset })),
                Value::Tile(t) => {
                    let strides = row_major(&t.shape);
                    Ok(Value::Tile(gather(&t, &[t.shape[1], t.shape[0]], &[strides[1], strides[0]], 0)))
                }
                other => Err(format!("cannot transpose {other:?}")),
            },
            ExprKind::Accessor { .. } | ExprKind::Lanes { .. } | ExprKind::Intrinsic { .. } => Err("backend constructs cannot be interpreted; interpret the portable body".into()),
            ExprKind::Builtin { name, args } => {
                if let Some(mut reduction) = crate::reduction::structured::Reduction::from_expr(e) {
                    // The reference interpreter's tiles are logical decoded
                    // values. Resolve generic types before constructing local
                    // state/leaf snapshots; physical packed planes belong to tensors.
                    for operand in reduction.operands_mut() {
                        if let Ty::Tile(shape)=&mut operand.ty {
                            let element=crate::lower::subst_elem(&shape.elem,&frame.elements);
                            shape.elem=Elem::Dtype(element.read_dtype().ok_or("unresolved reduction input element")?);
                            shape.packed_axis=None;
                        }
                    }
                    for call in std::iter::once(&mut reduction.merge).chain(reduction.step.iter_mut().map(|s|&mut s.call)) {
                        if let ExprKind::Call{elem_args,..}=&mut call.kind {
                            for element in elem_args {
                                let resolved=crate::lower::subst_elem(element,&frame.elements);
                                *element=Elem::Dtype(resolved.read_dtype().ok_or("unresolved reduction helper element")?);
                            }
                        }
                    }
                    let original = frame.vars.len();
                    let vars = self.vars_stack.last_mut().unwrap();
                    let body = reduction.expand(crate::reduction::structured::Tree::Ordered, vars)?;
                    frame.vars.resize(vars.len(), None);
                    let result = self.block(&body, frame);
                    frame.vars.truncate(original);
                    self.vars_stack.last_mut().unwrap().truncate(original);
                    result?;
                    Ok(Value::Void)
                } else { self.builtin(*name, args, frame) }
            }
            ExprKind::Call { callee, shape_args, elem_args, args } => {
                let f = self.program.functions.iter().find(|f| &f.name == callee).ok_or_else(|| format!("no function `{callee}`"))?.clone();
                let mut inner = Frame { vars: vec![None; f.vars.len()], index: HashMap::new(), shapes: HashMap::new(), elements: HashMap::new() };
                for (p, s) in f.shape_params.iter().zip(shape_args) {
                    inner.shapes.insert(p.clone(), self.eval_sym(s, frame));
                }
                for (p, element) in f.elem_params.iter().zip(elem_args) {
                    inner.elements.insert(p.clone(), crate::lower::subst_elem(element, &frame.elements));
                }
                // Tile arguments are passed by reference: copy in, run, copy back.
                let mut values = Vec::new();
                for a in args {
                    values.push(self.expr(a, frame)?);
                }
                for (i, v) in values.iter().enumerate() {
                    inner.vars[i] = Some(v.clone());
                }
                self.vars_stack.push(f.vars.clone());
                let result = self.block(&f.body, &mut inner);
                self.vars_stack.pop();
                result?;
                for (i, a) in args.iter().enumerate() {
                    if let ExprKind::Var(id) = a.kind {
                        if matches!(a.ty, Ty::Tile(_)) {
                            frame.vars[id] = inner.vars[i].clone();
                        }
                    }
                }
                Ok(Value::Void)
            }
            ExprKind::Unary { op, expr } => {
                let v = self.expr(expr, frame)?;
                Ok(match (op, v) {
                    (UnaryOp::Neg, Value::Scalar(x)) => Value::Scalar(-x),
                    (UnaryOp::Neg, Value::Int(x)) => Value::Int(crate::numeric::integer_value(scalar_dtype(&e.ty),(x as u32).wrapping_neg())),
                    (UnaryOp::Not, Value::Bool(b)) => Value::Bool(!b),
                    (UnaryOp::BitNot, Value::Int(x)) => Value::Int(crate::numeric::integer_value(scalar_dtype(&e.ty),!(x as u32))),
                    (op, v) => return Err(format!("unary {op:?} on {v:?}")),
                })
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.expr(lhs, frame)?;
                let r = self.expr(rhs, frame)?;
                self.binary(*op, l, r, &e.ty)
            }
            ExprKind::Cast { dtype, expr } => {
                let v = self.expr(expr, frame)?;
                if let Value::Int(x)=v {if dtype.is_int() {return Ok(Value::Int(crate::numeric::integer_value(*dtype,x as u32)));}}
                let x = match v {
                    Value::Scalar(x) => x,
                    Value::Int(x) => x as f64,
                    Value::Bool(b) => b as i64 as f64,
                    other => return Err(format!("cast of {other:?}")),
                };
                Ok(if dtype.is_int() { Value::Int(round_to(*dtype, x) as i64) } else { Value::Scalar(round_to(*dtype, x)) })
            }
            ExprKind::Tuple(items) => {
                let mut out = Vec::new();
                for i in items {
                    out.push(self.expr(i, frame)?);
                }
                Ok(Value::Tuple(out))
            }
        }
    }

    fn index_value(&self, item: Value, points: &[Option<usize>], slices: &[Option<(usize, usize)>]) -> Result<Value, String> {
        match item {
            Value::View(v) => {
                let mut ns = Vec::new();
                let mut nst = Vec::new();
                let mut off = v.offset;
                for (axis, extent) in v.shape.iter().enumerate() {
                    if axis < points.len() {
                        if let Some(p) = points[axis] {
                            if p >= *extent { return Err("point index outside view bounds".into()); }
                            off += p * v.strides[axis];
                        } else {
                            let (s, en) = slices[axis].unwrap();
                            let en = en.min(*extent);
                            let s = s.min(en);
                            off += s * v.strides[axis];
                            ns.push(en - s);
                            nst.push(v.strides[axis]);
                        }
                    } else {
                        ns.push(*extent);
                        nst.push(v.strides[axis]);
                    }
                }
                Ok(Value::View(View { tensor: v.tensor, shape: ns, strides: nst, offset: off }))
            }
            other => Err(format!("cannot index tuple element {other:?}")),
        }
    }

    fn binary(&self, op: BinaryOp, l: Value, r: Value, ty: &Ty) -> Result<Value, String> {
        match (l, r) {
            (Value::Scalar(a), Value::Scalar(b)) => {
                let d = scalar_dtype(ty);
                Ok(match op {
                    BinaryOp::Eq => Value::Bool(a == b),
                    BinaryOp::Ne => Value::Bool(a != b),
                    BinaryOp::Lt => Value::Bool(a < b),
                    BinaryOp::Le => Value::Bool(a <= b),
                    BinaryOp::Gt => Value::Bool(a > b),
                    BinaryOp::Ge => Value::Bool(a >= b),
                    _ => Value::Scalar(round_to(d, float_op(op, a, b)?)),
                })
            }
            (Value::Int(a), Value::Int(b)) => Ok(match op {
                BinaryOp::Add => Value::Int(crate::numeric::integer_value(scalar_dtype(ty),(a as u32).wrapping_add(b as u32))),
                BinaryOp::Sub => Value::Int(crate::numeric::integer_value(scalar_dtype(ty),(a as u32).wrapping_sub(b as u32))),
                BinaryOp::Mul => Value::Int(crate::numeric::integer_value(scalar_dtype(ty),(a as u32).wrapping_mul(b as u32))),
                BinaryOp::Div => Value::Int(checked_integer_division(a,b,ty,false)?),
                BinaryOp::Rem => Value::Int(checked_integer_division(a,b,ty,true)?),
                BinaryOp::Shl | BinaryOp::Shr => {
                    if !crate::numeric::integer_shift_is_defined(Some(b)) {return Err("integer shift count must be in 0..32".into());}
                    let bits=if op==BinaryOp::Shl {(a as u32).wrapping_shl(b as u32)}
                        else if *ty==Ty::Scalar(DType::I32) {((a as i32) >> b) as u32}
                        else {(a as u32) >> b};
                    Value::Int(crate::numeric::integer_value(scalar_dtype(ty),bits))
                },
                BinaryOp::BitAnd => Value::Int(a & b),
                BinaryOp::BitOr => Value::Int(a | b),
                BinaryOp::BitXor => Value::Int(a ^ b),
                BinaryOp::Eq => Value::Bool(a == b),
                BinaryOp::Ne => Value::Bool(a != b),
                BinaryOp::Lt => Value::Bool(a < b),
                BinaryOp::Le => Value::Bool(a <= b),
                BinaryOp::Gt => Value::Bool(a > b),
                BinaryOp::Ge => Value::Bool(a >= b),
                BinaryOp::And | BinaryOp::Or => return Err("logic on integers".into()),
            }),
            (Value::Scalar(a), Value::Int(b)) => self.binary(op, Value::Scalar(a), Value::Scalar(b as f64), ty),
            (Value::Int(a), Value::Scalar(b)) => self.binary(op, Value::Scalar(a as f64), Value::Scalar(b), ty),
            (Value::Bool(a), Value::Bool(b)) => Ok(match op {
                BinaryOp::And => Value::Bool(a && b),
                BinaryOp::Or => Value::Bool(a || b),
                BinaryOp::Eq => Value::Bool(a == b),
                BinaryOp::Ne => Value::Bool(a != b),
                _ => return Err("arithmetic on bools".into()),
            }),
            (Value::Tile(a), Value::Tile(b)) => {
                let Ty::Tile(s) = ty else { unreachable!() };
                let Elem::Dtype(d) = s.elem else { unreachable!() };
                if a.shape != b.shape { return Err("elementwise shape mismatch".into()); }
                let data = a.data.iter().zip(&b.data).map(|(x,y)| {
                    self.binary_element(op, scalar_value(a.dtype,*x), scalar_value(b.dtype,*y), d)
                }).collect::<Result<Vec<_>,_>>()?;
                Ok(Value::Tile(Tile { shape: a.shape, dtype: d, data }))
            }
            (Value::Tile(a), b @ (Value::Scalar(_) | Value::Int(_) | Value::Bool(_))) => {
                let Ty::Tile(s) = ty else { unreachable!() };
                let Elem::Dtype(d) = s.elem else { unreachable!() };
                let data = a.data.iter().map(|x| {
                    self.binary_element(op, scalar_value(a.dtype,*x), b.clone(), d)
                }).collect::<Result<Vec<_>,_>>()?;
                Ok(Value::Tile(Tile { shape: a.shape, dtype: d, data }))
            }
            (a @ (Value::Scalar(_) | Value::Int(_) | Value::Bool(_)), Value::Tile(b)) => {
                let Ty::Tile(s) = ty else { unreachable!() };
                let Elem::Dtype(d) = s.elem else { unreachable!() };
                let data = b.data.iter().map(|y| {
                    self.binary_element(op, a.clone(), scalar_value(b.dtype,*y), d)
                }).collect::<Result<Vec<_>,_>>()?;
                Ok(Value::Tile(Tile { shape: b.shape, dtype: d, data }))
            }
            (l, r) => Err(format!("binary {op:?} on {l:?} and {r:?}")),
        }
    }

    // Tile expressions have the same operation and rounding semantics as
    // evaluating the expression at each logical coordinate.
    fn binary_element(&self, op: BinaryOp, left: Value, right: Value, dtype: DType) -> Result<f64, String> {
        match self.binary(op,left,right,&Ty::Scalar(dtype))? {
            Value::Scalar(x) => Ok(x),
            Value::Int(x) => Ok(x as f64),
            Value::Bool(x) => Ok(u8::from(x) as f64),
            _ => Err("elementwise operation did not return a scalar".into()),
        }
    }

    fn builtin(&mut self, name: Builtin, args: &[Expr], frame: &mut Frame) -> Result<Value, String> {
        match name {
            Builtin::Reshape => {
                let Value::View(mut view) = self.expr(&args[0], frame)? else { return Err("reshape requires a tensor view".into()); };
                if matches!(self.tensors[view.tensor], TensorData::Packed { .. }) { return Err("reshape currently requires dense storage".into()); }
                // Evaluate dimensions after the source, even when checking has
                // proved their symbolic values. Nested extent queries can fail.
                for dimension in args.iter().skip(1) {
                    if !crate::effects::can_substitute_symbolic_value(dimension) {
                        self.scalar(dimension, frame)?;
                    }
                }
                let target = args[1..].iter().map(|e| e.sym.as_ref().map(|s| self.eval_sym(s,frame)).ok_or("reshape dimension is not symbolic")).collect::<Result<Vec<_>,_>>()?;
                let dims = view.shape.iter().map(|n| i64::try_from(*n).map_err(|_| "reshape extent overflow".to_string())).collect::<Result<Vec<_>,_>>()?;
                let strides = view.strides.iter().map(|n| i64::try_from(*n).map_err(|_| "reshape stride overflow".to_string())).collect::<Result<Vec<_>,_>>()?;
                let strides = crate::layout::reshape_strides(&dims,&strides,&target)?;
                view.shape = target.into_iter().map(|n| n as usize).collect();
                view.strides = strides.into_iter().map(|n| n as usize).collect();
                Ok(Value::View(view))
            }
            Builtin::Load => match self.expr(&args[0], frame)? {
                Value::View(v) => Ok(Value::Tile(self.materialize(&v))),
                Value::Tuple(items) => {
                    let mut out = Vec::new();
                    for i in items {
                        match i {
                            Value::View(v) => out.push(Value::Tile(self.materialize(&v))),
                            other => return Err(format!("load of {other:?}")),
                        }
                    }
                    Ok(Value::Tuple(out))
                }
                other => Err(format!("load of {other:?}")),
            },
            Builtin::Store => {
                let t = match self.expr(&args[0], frame)? {
                    Value::Tile(t) => t,
                    other => return Err(format!("store of {other:?}")),
                };
                let v = match self.expr(&args[1], frame)? {
                    Value::View(v) => v,
                    other => return Err(format!("store into {other:?}")),
                };
                if v.shape != t.shape {
                    return Err(format!("store shape mismatch {:?} vs {:?}", t.shape, v.shape));
                }
                let n: usize = v.shape.iter().product();
                let mut idx = vec![0usize; v.shape.len()];
                for k in 0..n {
                    let flat = v.flat(&idx);
                    self.tensors[v.tensor].set(flat, t.data[k]);
                    let mut a = idx.len();
                    while a > 0 {
                        a -= 1;
                        idx[a] += 1;
                        if idx[a] < v.shape[a] {
                            break;
                        }
                        idx[a] = 0;
                    }
                }
                Ok(Value::Void)
            }
            Builtin::Atomic => {
                let v = match self.expr(&args[0], frame)? {
                    Value::View(v) => v,
                    other => return Err(format!("atomic on {other:?}")),
                };
                let x = match self.expr(&args[1], frame)? {
                    Value::Scalar(x) => x,
                    Value::Int(x) => x as f64,
                    other => return Err(format!("atomic value {other:?}")),
                };
                let cur = self.tensors[v.tensor].get(v.offset);
                self.tensors[v.tensor].set(v.offset, cur + x);
                Ok(Value::Void)
            }
            Builtin::Reduce => {
                let t = match self.expr(&args[0], frame)? {
                    Value::Tile(t) => t,
                    other => return Err(format!("reduce of {other:?}")),
                };
                let axis = match self.expr(&args[1], frame)? {
                    Value::Int(a) => a as usize,
                    _ => unreachable!(),
                };
                let ExprKind::Int(op) = args[2].kind else { unreachable!() };
                let op = ReduceOp::from_tag(op).ok_or("invalid reduction operation")?;
                let contract = crate::reduction::Contract::new(op, t.dtype,
                    matches!(args.get(3).map(|e| &e.kind), Some(ExprKind::Bool(true))));
                let mut out_shape = t.shape.clone();
                out_shape.remove(axis);
                let n: usize = out_shape.iter().product();
                let mut out = vec![0f64; n];
                let inner: usize = t.shape[axis + 1..].iter().product();
                let outer: usize = t.shape[..axis].iter().product();
                let extent = t.shape[axis];
                if !contract.allows_empty_axis() && extent==0 {return Err("argmax requires a nonempty axis".into());}
                let dtype = contract.output();
                for o in 0..outer {
                    for i in 0..inner {
                        let mut acc = contract.identity().value();
                        let mut arg = 0usize;
                        for k in 0..extent {
                            let x = t.data[(o * extent + k) * inner + i];
                            match op {
                                ReduceOp::Sum => acc = round_to(t.dtype, acc + x),
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
                        out[o * inner + i] = if op == ReduceOp::Argmax { arg as f64 } else { acc };
                    }
                }
                if out_shape.is_empty() {
                    Ok(scalar_value(dtype,out[0]))
                } else {
                    Ok(Value::Tile(Tile { shape: out_shape, dtype, data: out }))
                }
            }
            Builtin::Extent => {
                let t = self.expr(&args[0], frame)?;
                let axis = match self.expr(&args[1], frame)? {
                    Value::Int(a) => a as usize,
                    _ => unreachable!(),
                };
                Ok(Value::Int(match t {
                    Value::Tile(t) => t.shape[axis] as i64,
                    Value::View(v) => v.shape[axis] as i64,
                    other => return Err(format!("extent of {other:?}")),
                }))
            }
            Builtin::Fma => {
                let d = scalar_dtype(&args[0].ty);
                let a = self.scalar(&args[0], frame)?;
                let b = self.scalar(&args[1], frame)?;
                let c = self.scalar(&args[2], frame)?;
                Ok(Value::Scalar(round_to(d, a.mul_add(b, c))))
            }
            Builtin::Exp | Builtin::ExpFast => {
                let d = scalar_dtype(&args[0].ty);
                let a = self.scalar(&args[0], frame)?;
                Ok(Value::Scalar(round_to(d, a.exp())))
            }
            Builtin::Rsqrt => {
                let d = scalar_dtype(&args[0].ty);
                let a = self.scalar(&args[0], frame)?;
                Ok(Value::Scalar(round_to(d, 1.0 / a.sqrt())))
            }
            Builtin::Log | Builtin::Sin | Builtin::Cos => {
                let d = scalar_dtype(&args[0].ty);
                let a = self.scalar(&args[0], frame)?;
                let r = match name {
                    Builtin::Log => a.ln(),
                    Builtin::Sin => a.sin(),
                    _ => a.cos(),
                };
                Ok(Value::Scalar(round_to(d, r)))
            }
            Builtin::Sqrt => {
                let d = scalar_dtype(&args[0].ty);
                let a = self.scalar(&args[0], frame)?;
                Ok(Value::Scalar(round_to(d, a.sqrt())))
            }
            Builtin::Abs => {
                let d = scalar_dtype(&args[0].ty);
                let a = self.scalar(&args[0], frame)?;
                Ok(if d.is_int() { Value::Int(if d==DType::I32 {i64::from((a as i32).wrapping_abs())}else{a as i64}) } else { Value::Scalar(round_to(d, a.abs())) })
            }
            Builtin::Max | Builtin::Min => {
                let a = self.scalar(&args[0], frame)?;
                let b = self.scalar(&args[1], frame)?;
                let r = if name == Builtin::Max { a.max(b) } else { a.min(b) };
                Ok(if scalar_dtype(&args[0].ty).is_int() { Value::Int(r as i64) } else { Value::Scalar(r) })
            }
        }
    }

    fn scalar(&mut self, e: &Expr, frame: &mut Frame) -> Result<f64, String> {
        match self.expr(e, frame)? {
            Value::Scalar(x) => Ok(x),
            Value::Int(x) => Ok(x as f64),
            other => Err(format!("expected a scalar, found {other:?}")),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Arg {
    Tensor(usize),
    Scalar(f64),
}

fn full_view(id: usize, shape: &[usize]) -> View {
    View { tensor: id, shape: shape.to_vec(), strides: row_major(shape), offset: 0 }
}

fn row_major(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn gather(t: &Tile, shape: &[usize], strides: &[usize], offset: usize) -> Tile {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..n {
        let flat = offset + idx.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>();
        data.push(t.data[flat]);
        let mut k = idx.len();
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            if idx[k] < shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    Tile { shape: shape.to_vec(), dtype: t.dtype, data }
}

fn scalar_dtype(ty: &Ty) -> DType {
    match ty {
        Ty::Scalar(d) => *d,
        _ => DType::F32,
    }
}

fn scalar_value(dtype: DType, value: f64) -> Value {
    if dtype == DType::Bool { Value::Bool(value != 0.0) }
    else if dtype.is_int() { Value::Int(value as i64) }
    else { Value::Scalar(value) }
}
fn integer_arith(op: AssignOp, dtype: DType, a: i64, b: i64) -> i64 {
    let (a,b) = (a as u32, b as u32);
    let bits = match op {
        AssignOp::Assign => b,
        AssignOp::Add => a.wrapping_add(b),
        AssignOp::Sub => a.wrapping_sub(b),
        AssignOp::Mul => a.wrapping_mul(b),
    };
    crate::numeric::integer_value(dtype,bits)
}
fn typed_arith(op: AssignOp, dtype: DType, a: f64, b: f64) -> f64 {
    if dtype.is_int() { integer_arith(op,dtype,a as i64,b as i64) as f64 }
    else { round_to(dtype,arith(op,a,b)) }
}
fn arith(op: AssignOp, a: f64, b: f64) -> f64 {
    match op {
        AssignOp::Assign => b,
        AssignOp::Add => a + b,
        AssignOp::Sub => a - b,
        AssignOp::Mul => a * b,
    }
}

fn float_op(op: BinaryOp, a: f64, b: f64) -> Result<f64, String> {
    Ok(match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Rem => a % b,
        other => return Err(format!("{other:?} on floats")),
    })
}

/// Deterministic pseudo-random numbers for test inputs.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() % 1_000_000) as f64 / 1_000_000.0
    }
}

impl TensorData {
    /// A dense tensor with values uniform in [-1, 1), rounded to the dtype.
    pub fn random_dense(rng: &mut Rng, dtype: DType, shape: Vec<usize>) -> TensorData {
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| round_to(dtype, rng.unit() * 2.0 - 1.0)).collect();
        TensorData::Dense { dtype, shape, data }
    }

    /// A packed tensor with random codes and small positive scales.
    pub fn random_packed(rng: &mut Rng, rep: &'static repr::Repr, shape: Vec<usize>) -> TensorData {
        let k = *shape.last().unwrap();
        let rows: usize = shape[..shape.len() - 1].iter().product();
        assert!(k % rep.storage_group() as usize == 0, "packed rows require complete storage groups");
        let count = rows * k;
        let mut planes = Vec::new();
        for plane in rep.planes() {
            let mut bytes = vec![0; plane.bytes(count as u64).unwrap() as usize];
            let entries = plane.entries(count as u64).unwrap() as usize;
            match plane.encoding {
                repr::PlaneEncoding::Packed { bits, .. } => {
                    for entry in 0..entries { repr::write_packed(&mut bytes, entry, bits, rng.next() as u32); }
                }
                repr::PlaneEncoding::Dense(dtype) => {
                    for entry in 0..entries {
                        let value = if plane.name == "bias" { (rng.unit() - 0.5) as f32 } else { (rng.unit() * 0.01 + 0.001) as f32 };
                        let start = entry * dtype.bytes() as usize;
                        match dtype {
                            DType::F32 => bytes[start..start+4].copy_from_slice(&value.to_le_bytes()),
                            DType::F16 => bytes[start..start+2].copy_from_slice(&f16_bits(value).to_le_bytes()),
                            DType::BF16 => bytes[start..start+2].copy_from_slice(&((bf16_round(value).to_bits() >> 16) as u16).to_le_bytes()),
                            _ => unreachable!("nonfloating coefficient"),
                        }
                    }
                }
            }
            planes.push(bytes);
        }
        TensorData::Packed { repr: rep, shape, planes }
    }

    /// Byte images of the buffers this tensor occupies on a device, in ABI order.
    pub fn device_bytes(&self) -> Vec<Vec<u8>> {
        match self {
            TensorData::Dense { dtype, data, .. } => {
                let mut out = Vec::with_capacity(data.len() * dtype.bytes() as usize);
                for v in data {
                    match dtype {
                        DType::F32 => out.extend_from_slice(&(*v as f32).to_le_bytes()),
                        DType::BF16 => out.extend_from_slice(&((*v as f32).to_bits() >> 16).to_le_bytes()[..2]),
                        DType::F16 => out.extend_from_slice(&f16_bits(*v as f32).to_le_bytes()),
                        DType::I32 => out.extend_from_slice(&(*v as i32).to_le_bytes()),
                        DType::U32 => out.extend_from_slice(&(*v as u32).to_le_bytes()),
                        DType::Bool => out.push(*v as u8),
                    }
                }
                vec![out]
            }
            TensorData::Packed { planes, .. } => planes.clone(),
        }
    }

    /// Replace a dense tensor's values from device bytes.
    pub fn load_device_bytes(&mut self, bytes: &[u8]) {
        let TensorData::Dense { dtype, data, .. } = self else { panic!("only dense tensors are read back") };
        let w = dtype.bytes() as usize;
        for (i, v) in data.iter_mut().enumerate() {
            let b = &bytes[i * w..(i + 1) * w];
            *v = match dtype {
                DType::F32 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
                DType::BF16 => f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16) as f64,
                DType::F16 => f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())) as f64,
                DType::I32 => i32::from_le_bytes(b.try_into().unwrap()) as f64,
                DType::U32 => u32::from_le_bytes(b.try_into().unwrap()) as f64,
                DType::Bool => b[0] as f64,
            };
        }
    }
}

fn checked_integer_division(a:i64,b:i64,ty:&Ty,remainder:bool)->Result<i64,String>{
    if !crate::numeric::integer_division_is_defined(scalar_dtype(ty),Some(a),Some(b)) { return Err("integer division by zero or signed overflow".into()); }
    (if remainder {a.checked_rem_euclid(b)} else {a.checked_div_euclid(b)}).ok_or_else(||"integer division overflow".into())
}
