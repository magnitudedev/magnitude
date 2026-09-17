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
use crate::hir::*;
use crate::program::Program;
use crate::repr;
use crate::sym::{Atom, Sym};
use crate::types::{DType, Elem, Ty};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum TensorData {
    Dense { dtype: DType, shape: Vec<usize>, data: Vec<f64> },
    /// Packed along the last axis: `words` holds `codes_per_word` codes each; coefficients per group.
    Packed { repr: &'static repr::Repr, shape: Vec<usize>, words: Vec<u32>, scale: Vec<f32>, bias: Vec<f32> },
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
            TensorData::Packed { repr, shape, words, scale, bias } => {
                let k = *shape.last().unwrap();
                let row = flat / k;
                let col = flat % k;
                let cpw = repr.codes_per_word() as usize;
                let words_per_row = k / cpw;
                let word = words[row * words_per_row + col / cpw];
                let code = (word >> ((col % cpw) as u32 * repr.bits)) & ((1u32 << repr.bits) - 1);
                let groups_per_row = k / repr.group as usize;
                let g = row * groups_per_row + col / repr.group as usize;
                let b = if repr.has_bias { bias[g] } else { 0.0 };
                (scale[g] as f64 * repr.decode_code(code) as f64 + b as f64) as f32 as f64
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
            TensorData::Packed { words, scale, bias, .. } => words.len() * 4 + scale.len() * 4 + bias.len() * 4,
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
    /// A view of a tile: indices into a tile variable with a mapping.
    TileView { tile: usize, shape: Vec<usize>, strides: Vec<usize>, offset: usize },
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
            StmtKind::Parallel { vars, extents, body } => {
                let ext: Vec<i64> = extents.iter().map(|e| self.eval_sym(e, frame)).collect();
                let names = self.index_names(vars);
                self.nested(vars, &ext, body, frame, &names)
            }
            StmtKind::Owned { vars, tile, body } => {
                let t = self.expr(tile, frame)?;
                let shape: Vec<i64> = match &t {
                    Value::Tile(t) => t.shape.iter().map(|d| *d as i64).collect(),
                    Value::TileView { shape, .. } => shape.iter().map(|d| *d as i64).collect(),
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
            StmtKind::LoadLoop { vars, views, axis, piece, body, .. } => {
                // The interpreter takes the whole axis as one piece; pieces are a lowering choice.
                let mut tiles = Vec::new();
                let mut extent = 0;
                for v in views {
                    let view = match self.expr(v, frame)? {
                        Value::View(v) => v,
                        other => return Err(format!("load of non-view {other:?}")),
                    };
                    extent = view.shape[*axis];
                    tiles.push(self.materialize(&view));
                }
                if extent == 0 { return Ok(()); }
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
                            (op, Some(Value::Int(a)), Value::Int(b)) => Value::Int(arith(*op, a as f64, *b as f64) as i64),
                            (op, Some(Value::Tile(mut t)), Value::Tile(b)) => {
                                for (x, y) in t.data.iter_mut().zip(&b.data) {
                                    *x = round_to(t.dtype, arith(*op, *x, *y));
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
                            _ => arith(*op, cur, scalar),
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
                    if indices.iter().all(|i| matches!(i, Index::Point(_))) {
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
                                return Ok(if t.dtype.is_int() { Value::Int(x as i64) } else { Value::Scalar(x) });
                            }
                            Some(Value::View(v)) if idx.len() == v.shape.len() => {
                                if idx.iter().zip(&v.shape).any(|(i,n)| i >= n) { return Err("point index outside view bounds".into()); }
                                let flat = v.flat(&idx);
                                let x = self.tensors[v.tensor].get(flat);
                                let is_int = matches!(&self.tensors[v.tensor], TensorData::Dense { dtype, .. } if dtype.is_int());
                                return Ok(if is_int { Value::Int(x as i64) } else { Value::Scalar(x) });
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
                        if shape.is_empty() {
                            let x = self.tensors[v.tensor].get(offset);
                            let is_int = matches!(&self.tensors[v.tensor], TensorData::Dense { dtype, .. } if dtype.is_int());
                            Ok(if is_int { Value::Int(x as i64) } else { Value::Scalar(x) })
                        } else {
                            Ok(Value::View(View { tensor: v.tensor, shape, strides, offset }))
                        }
                    }
                    Value::Tile(t) => {
                        let strides = row_major(&t.shape);
                        let (shape, strides, offset) = apply(&t.shape, &strides, 0)?;
                        if shape.is_empty() {
                            return Ok(if t.dtype.is_int() { Value::Int(t.data[offset] as i64) } else { Value::Scalar(t.data[offset]) });
                        }
                        Ok(Value::Tile(gather(&t, &shape, &strides, offset)))
                    }
                    Value::TileView { .. } => Err("nested tile views are not supported".into()),
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
            ExprKind::Builtin { name, args } => self.builtin(*name, args, frame),
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
                    (UnaryOp::Neg, Value::Int(x)) => Value::Int(-x),
                    (UnaryOp::Not, Value::Bool(b)) => Value::Bool(!b),
                    (UnaryOp::BitNot, Value::Int(x)) => Value::Int(!x),
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
                BinaryOp::Add => Value::Int(a + b),
                BinaryOp::Sub => Value::Int(a - b),
                BinaryOp::Mul => Value::Int(a * b),
                BinaryOp::Div => Value::Int(checked_integer_division(a,b,ty,false)?),
                BinaryOp::Rem => Value::Int(checked_integer_division(a,b,ty,true)?),
                BinaryOp::Shl | BinaryOp::Shr => {
                    if !(0..32).contains(&b) {return Err("integer shift count must be in 0..32".into());}
                    let bits=if op==BinaryOp::Shl {(a as u32).wrapping_shl(b as u32)}
                        else if *ty==Ty::Scalar(DType::I32) {((a as i32) >> b) as u32}
                        else {(a as u32) >> b};
                    Value::Int(if *ty==Ty::Scalar(DType::I32) {i64::from(bits as i32)}else{i64::from(bits)})
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
                let data = a.data.iter().zip(&b.data).map(|(x, y)| float_op(op, *x, *y).map(|v| round_to(d, v))).collect::<Result<Vec<_>, _>>()?;
                Ok(Value::Tile(Tile { shape: a.shape, dtype: d, data }))
            }
            (Value::Tile(a), Value::Scalar(b)) | (Value::Scalar(b), Value::Tile(a)) => {
                let Ty::Tile(s) = ty else { unreachable!() };
                let Elem::Dtype(d) = s.elem else { unreachable!() };
                let data = a.data.iter().map(|x| float_op(op, *x, b).map(|v| round_to(d, v))).collect::<Result<Vec<_>, _>>()?;
                Ok(Value::Tile(Tile { shape: a.shape, dtype: d, data }))
            }
            (l, r) => Err(format!("binary {op:?} on {l:?} and {r:?}")),
        }
    }

    fn builtin(&mut self, name: Builtin, args: &[Expr], frame: &mut Frame) -> Result<Value, String> {
        match name {
            Builtin::Reshape => {
                let Value::View(mut view) = self.expr(&args[0], frame)? else { return Err("reshape requires a tensor view".into()); };
                if matches!(self.tensors[view.tensor], TensorData::Packed { .. }) { return Err("reshape currently requires dense storage".into()); }
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
                let op = match op {
                    0 => ReduceOp::Sum,
                    1 => ReduceOp::Max,
                    2 => ReduceOp::Min,
                    _ => ReduceOp::Argmax,
                };
                let mut out_shape = t.shape.clone();
                out_shape.remove(axis);
                let n: usize = out_shape.iter().product();
                let mut out = vec![0f64; n];
                let inner: usize = t.shape[axis + 1..].iter().product();
                let outer: usize = t.shape[..axis].iter().product();
                let extent = t.shape[axis];
                if op==ReduceOp::Argmax && extent==0 {return Err("argmax requires a nonempty axis".into());}
                let dtype = if op == ReduceOp::Argmax { DType::I32 } else { t.dtype };
                for o in 0..outer {
                    for i in 0..inner {
                        let mut acc = match op {
                            ReduceOp::Sum => 0.0,
                            ReduceOp::Max => f64::NEG_INFINITY,
                            ReduceOp::Min => f64::INFINITY,
                            ReduceOp::Argmax => f64::NEG_INFINITY,
                        };
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
                    Ok(if dtype.is_int() { Value::Int(out[0] as i64) } else { Value::Scalar(out[0]) })
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
                Ok(if d.is_int() { Value::Int(a.abs() as i64) } else { Value::Scalar(round_to(d, a.abs())) })
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
        let cpw = rep.codes_per_word() as usize;
        assert!(k % cpw == 0 && k % rep.group as usize == 0, "packed extent must be a multiple of the packet and group");
        let mut words = vec![0u32; rows * k / cpw];
        let groups = rows * k / rep.group as usize;
        let coeff = |x: f32| round_to(rep.coefficient, x as f64) as f32;
        let scale: Vec<f32> = (0..groups).map(|_| coeff((rng.unit() * 0.1 + 0.01) as f32)).collect();
        let bias: Vec<f32> = (0..groups).map(|_| coeff((rng.unit() - 0.5) as f32)).collect();
        for w in words.iter_mut() {
            let mut v = 0u32;
            for c in 0..cpw {
                v |= ((rng.next() as u32) & ((1 << rep.bits) - 1)) << (c as u32 * rep.bits);
            }
            *w = v;
        }
        TensorData::Packed { repr: rep, shape, words, scale, bias: if rep.has_bias { bias } else { Vec::new() } }
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
            TensorData::Packed { words, scale, bias, repr, .. } => {
                let mut w = Vec::with_capacity(words.len() * 4);
                for x in words {
                    w.extend_from_slice(&x.to_le_bytes());
                }
                let coeff_bytes = |values: &Vec<f32>| -> Vec<u8> {
                    let mut out = Vec::with_capacity(values.len() * repr.coefficient.bytes() as usize);
                    for x in values {
                        match repr.coefficient {
                            DType::BF16 => out.extend_from_slice(&(x.to_bits() >> 16).to_le_bytes()[..2]),
                            DType::F16 => out.extend_from_slice(&f16_bits(*x).to_le_bytes()),
                            _ => out.extend_from_slice(&x.to_le_bytes()),
                        }
                    }
                    out
                };
                let mut out = vec![w, coeff_bytes(scale)];
                if repr.has_bias {
                    out.push(coeff_bytes(bias));
                }
                out
            }
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
    if b==0 || (matches!(ty,Ty::Scalar(DType::I32)) && a==i64::from(i32::MIN) && b == -1) { return Err("integer division by zero or signed overflow".into()); }
    (if remainder {a.checked_rem_euclid(b)} else {a.checked_div_euclid(b)}).ok_or_else(||"integer division overflow".into())
}
