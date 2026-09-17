//! Demand: what a function's portable body requires of the machine, for concrete shapes.
//!
//! Bytes are counted per tensor as the smaller of the tensor's size and the bytes its views
//! are touched over every loop iteration: a tensor swept once costs its size, a tensor re-read
//! by every work item costs its size once. Operations are counted from the loop structure.
//! Data-dependent extents are taken at their upper bound, so the demand is an upper bound
//! wherever the data decides.

use seismic_lang::hir::*;
use seismic_lang::program::Program;
use seismic_lang::sym::{Atom, Sym};
use seismic_lang::types::{Elem, Ty};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct Demand {
    pub bytes_read: f64,
    pub bytes_written: f64,
    pub flops: f64,
    pub per_tensor: Vec<(String, f64, f64)>,
}

impl Demand {
    pub fn bytes(&self) -> f64 {
        self.bytes_read + self.bytes_written
    }
}

struct Walker<'a> {
    program: &'a Program,
    shapes: HashMap<String, i64>,
    /// touched (read, written) bytes per parameter tensor, summed over iterations
    touched: HashMap<String, (f64, f64)>,
    flops: f64,
}

pub fn demand(program: &Program, name: &str, shapes: &HashMap<String, i64>) -> Result<Demand, String> {
    let f = program.functions.iter().find(|f| f.name == name).ok_or_else(|| format!("no function `{name}`"))?;
    let mut w = Walker { program, shapes: shapes.clone(), touched: HashMap::new(), flops: 0.0 };
    let mut env: HashMap<String, f64> = shapes.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();
    let params: HashMap<VarId, String> = f.params.iter().enumerate().map(|(i, (n, _))| (i, n.clone())).collect();
    w.block(&f.body, f, &params, &mut env, 1.0)?;
    let mut d = Demand::default();
    for (pname, pty) in &f.params {
        let Ty::Tensor(s) = pty else { continue };
        let elems: f64 = s.shape.iter().map(|d| w.eval(d, &env)).product();
        let bits = match &s.elem {
            Elem::Dtype(d) => d.bytes() as f64 * 8.0,
            Elem::Repr(r) => seismic_lang::repr::lookup(r).unwrap().bits_per_value(),
            Elem::Param(_) => 32.0,
        };
        let size = elems * bits / 8.0;
        let (r, wr) = w.touched.get(pname).copied().unwrap_or((0.0, 0.0));
        let r = r.min(size);
        let wr = wr.min(size);
        d.bytes_read += r;
        d.bytes_written += wr;
        d.per_tensor.push((pname.clone(), r, wr));
    }
    d.flops = w.flops;
    Ok(d)
}

impl<'a> Walker<'a> {
    fn eval(&self, s: &Sym, env: &HashMap<String, f64>) -> f64 {
        s.eval(&|p| env.get(p).map(|v| *v as i64)).map(|v| v as f64).unwrap_or(f64::NAN)
    }

    fn block(&mut self, stmts: &[Stmt], f: &Function, params: &HashMap<VarId, String>, env: &mut HashMap<String, f64>, mult: f64) -> Result<(), String> {
        for s in stmts {
            self.stmt(s, f, params, env, mult)?;
        }
        Ok(())
    }

    fn atom_name(f: &Function, v: VarId) -> String {
        match &f.vars[v].kind {
            VarKind::Index(Atom::Param(p)) => p.clone(),
            _ => String::new(),
        }
    }

    fn stmt(&mut self, s: &Stmt, f: &Function, params: &HashMap<VarId, String>, env: &mut HashMap<String, f64>, mult: f64) -> Result<(), String> {
        match &s.kind {
            StmtKind::Parallel { vars, extents, body } => {
                let mut m = mult;
                for (v, e) in vars.iter().zip(extents) {
                    let n = self.eval(e, env);
                    m *= n;
                    // Loop indices take a representative value; extents drive the multiplier.
                    env.insert(Self::atom_name(f, *v), 0.0);
                }
                self.block(body, f, params, env, m)
            }
            StmtKind::Owned { vars, tile, body } => {
                let Ty::Tile(sh) = &tile.ty else { return Ok(()) };
                let mut m = mult;
                for (v, d) in vars.iter().zip(&sh.shape) {
                    m *= self.eval(d, env);
                    env.insert(Self::atom_name(f, *v), 0.0);
                }
                self.block(body, f, params, env, m)
            }
            StmtKind::Range { var, lo, hi, body } => {
                let n = (self.eval(hi, env) - self.eval(lo, env)).max(0.0);
                env.insert(Self::atom_name(f, *var), 0.0);
                self.block(body, f, params, env, mult * n)
            }
            StmtKind::Lanes { var, extent, body, .. } => {
                let n = self.eval(extent, env);
                env.insert(Self::atom_name(f, *var), 0.0);
                self.block(body, f, params, env, mult * n)
            }
            StmtKind::LoadLoop { vars, views, axis, piece, body, .. } => {
                // Streaming a view reads it once, in pieces; the piece extent is the whole axis.
                let mut extent = 0.0;
                for v in views {
                    if let Ty::Tensor(s) = &v.ty {
                        extent = self.upper(&s.shape[*axis], env);
                    }
                    self.touch_view(v, params, env, mult, true)?;
                }
                let Atom::Param(p) = piece else { unreachable!() };
                env.insert(p.clone(), extent);
                for v in vars {
                    let _ = v;
                }
                self.block(body, f, params, env, mult)
            }
            StmtKind::If { then, els, .. } => {
                // Both branches are possible; count the larger.
                let before = (self.flops, self.touched.clone());
                self.block(then, f, params, env, mult)?;
                let after_then = (self.flops, self.touched.clone());
                self.flops = before.0;
                self.touched = before.1;
                self.block(els, f, params, env, mult)?;
                if after_then.0 > self.flops {
                    self.flops = after_then.0;
                    self.touched = after_then.1;
                }
                Ok(())
            }
            StmtKind::Assign { value, .. } => self.expr(value, params, env, mult),
            StmtKind::Expr(e) => self.expr(e, params, env, mult),
        }
    }

    /// Upper bound of an extent: dynamic atoms are unknown here; the caller has bound them
    /// to their axis extents when they were introduced.
    fn upper(&self, s: &Sym, env: &HashMap<String, f64>) -> f64 {
        let v = self.eval(s, env);
        if v.is_nan() { 0.0 } else { v }
    }

    fn expr(&mut self, e: &Expr, params: &HashMap<VarId, String>, env: &mut HashMap<String, f64>, mult: f64) -> Result<(), String> {
        match &e.kind {
            ExprKind::Call { callee, shape_args, args, .. } => {
                let g = self.program.functions.iter().find(|f| &f.name == callee).ok_or_else(|| format!("no function `{callee}`"))?.clone();
                let mut inner: HashMap<String, f64> = HashMap::new();
                for (p, s) in g.shape_params.iter().zip(shape_args) {
                    inner.insert(p.clone(), self.eval(s, env));
                }
                // Arguments that are tensor views are touched by the callee once per call.
                let mut inner_params: HashMap<VarId, String> = HashMap::new();
                for (i, a) in args.iter().enumerate() {
                    if let Some(name) = self.view_param(a, params) {
                        inner_params.insert(i, name);
                    }
                }
                self.block(&g.body, &g, &inner_params, &mut inner, mult)
            }
            ExprKind::Builtin { name, args } => {
                match name {
                    Builtin::Load => {
                        let items = match &args[0].kind {
                            ExprKind::Tuple(items) => items.iter().collect::<Vec<_>>(),
                            _ => vec![&args[0]],
                        };
                        for v in items {
                            self.touch_view(v, params, env, mult, true)?;
                        }
                    }
                    Builtin::Store => self.touch_view(&args[1], params, env, mult, false)?,
                    Builtin::Atomic => self.touch_view(&args[0], params, env, mult, false)?,
                    Builtin::Fma => self.flops += 2.0 * mult,
                    Builtin::Exp | Builtin::ExpFast | Builtin::Rsqrt | Builtin::Sqrt | Builtin::Log | Builtin::Sin | Builtin::Cos => self.flops += 8.0 * mult,
                    Builtin::Reduce => {
                        if let Ty::Tile(sh) = &args[0].ty {
                            let n: f64 = sh.shape.iter().map(|d| self.eval(d, env)).product();
                            self.flops += n * mult;
                        }
                    }
                    _ => {}
                }
                for a in args {
                    self.expr(a, params, env, mult)?;
                }
                Ok(())
            }
            ExprKind::Binary { op, lhs, rhs } => {
                if matches!(lhs.ty, Ty::Scalar(d) if d.is_float()) && matches!(op, seismic_lang::ast::BinaryOp::Add | seismic_lang::ast::BinaryOp::Sub | seismic_lang::ast::BinaryOp::Mul | seismic_lang::ast::BinaryOp::Div) {
                    self.flops += mult;
                }
                self.expr(lhs, params, env, mult)?;
                self.expr(rhs, params, env, mult)
            }
            ExprKind::Index { base, indices } => {
                // An element read of a parameter tensor inside a loop is a touch of one element.
                if let Some(name) = self.view_param(base, params) {
                    if let Ty::Scalar(_) = e.ty {
                        if let Ty::Tensor(_) | Ty::Tile(_) = base.ty {
                            let bytes = self.elem_bytes(&base.ty);
                            let entry = self.touched.entry(name).or_insert((0.0, 0.0));
                            entry.0 += bytes * mult;
                        }
                    }
                }
                for i in indices {
                    match i {
                        Index::Point(p) => self.expr(p, params, env, mult)?,
                        Index::Slice { start, end } => {
                            if let Some(x) = start {
                                self.expr(x, params, env, mult)?;
                            }
                            if let Some(x) = end {
                                self.expr(x, params, env, mult)?;
                            }
                        }
                    }
                }
                self.expr(base, params, env, mult)
            }
            ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } | ExprKind::Transpose(expr) => self.expr(expr, params, env, mult),
            ExprKind::Tuple(items) => {
                for i in items {
                    self.expr(i, params, env, mult)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn elem_bytes(&self, ty: &Ty) -> f64 {
        match ty.shaped().map(|s| &s.elem) {
            Some(Elem::Dtype(d)) => d.bytes() as f64,
            Some(Elem::Repr(r)) => seismic_lang::repr::lookup(r).unwrap().bits_per_value() / 8.0,
            _ => 4.0,
        }
    }

    /// The parameter tensor a view expression refers to, if any.
    fn view_param(&self, e: &Expr, params: &HashMap<VarId, String>) -> Option<String> {
        match &e.kind {
            ExprKind::Var(v) => params.get(v).cloned(),
            ExprKind::Index { base, .. } | ExprKind::Transpose(base) | ExprKind::Accessor { base, .. } => self.view_param(base, params),
            _ => None,
        }
    }

    /// Account a view's bytes per iteration, with dynamic slice extents at their upper bound.
    fn touch_view(&mut self, v: &Expr, params: &HashMap<VarId, String>, env: &mut HashMap<String, f64>, mult: f64, read: bool) -> Result<(), String> {
        if let ExprKind::Tuple(items) = &v.kind {
            for i in items {
                self.touch_view(i, params, env, mult, read)?;
            }
            return Ok(());
        }
        let Some(name) = self.view_param(v, params) else { return Ok(()) };
        let Ty::Tensor(s) = &v.ty else { return Ok(()) };
        self.bind_dynamic_extents(v, env);
        let elems: f64 = s.shape.iter().map(|d| self.upper(d, env)).product();
        let bytes = elems * self.elem_bytes(&v.ty) * mult;
        let entry = self.touched.entry(name).or_insert((0.0, 0.0));
        if read {
            entry.0 += bytes;
        } else {
            entry.1 += bytes;
        }
        Ok(())
    }

    /// A dynamic slice's extent atom is bounded by the sliced axis of its base.
    fn bind_dynamic_extents(&self, v: &Expr, env: &mut HashMap<String, f64>) {
        if let ExprKind::Index { base, indices } = &v.kind {
            if let (Ty::Tensor(bs), Ty::Tensor(rs)) = (&base.ty, &v.ty) {
                let mut out_axis = 0;
                for (axis, idx) in indices.iter().enumerate() {
                    if let Index::Slice { .. } = idx {
                        let ext = &rs.shape[out_axis];
                        if ext.as_constant().is_none() {
                            if let [Atom::Param(p)] = ext.atoms().as_slice() {
                                if !env.contains_key(p) {
                                    let bound = self.eval(&bs.shape[axis], env);
                                    env.insert(p.clone(), bound);
                                }
                            }
                        }
                        out_axis += 1;
                    }
                }
                for axis in indices.len()..bs.shape.len() {
                    let _ = axis;
                }
            }
            self.bind_dynamic_extents(base, env);
        }
    }
}
