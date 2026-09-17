//! Launch planning for composition functions: a `fn` whose body is kernel calls, `range`
//! loops and `if`s over static integers. The plan is the unrolled sequence of kernel
//! invocations with concrete shapes, every tensor argument resolved to a contiguous view
//! of a caller parameter, and every scalar argument to a literal or a caller scalar.

use crate::hir::{Expr, ExprKind, Function, Index, Stmt, StmtKind, VarKind};
use crate::program::Program;
use crate::sym::{Atom, Sym};
use crate::types::{Elem, Ty};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Plan {
    pub function: String,
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub kernel: String,
    /// The callee's shape parameters, concrete.
    pub shapes: HashMap<String, i64>,
    /// One per tensor parameter of the callee, in parameter order.
    pub tensors: Vec<TensorBinding>,
    /// One per scalar parameter of the callee, in parameter order.
    pub scalars: Vec<(String, ScalarSource)>,
}

#[derive(Clone, Debug)]
pub struct TensorBinding {
    /// Callee parameter name.
    pub param: String,
    /// Caller parameter the view lives in.
    pub root: String,
    /// Offset of the view's first element, in elements of the root tensor.
    pub elem_offset: i64,
    /// The view's element type (same as the root's).
    pub elem: Elem,
}

#[derive(Clone, Debug)]
pub enum ScalarSource {
    Literal(f64),
    /// A scalar parameter of the composition function, supplied at run time.
    Param(String),
}

struct Planner<'a> {
    program: &'a Program,
    f: &'a Function,
    env: HashMap<String, i64>,
    /// Concrete shapes of the composition function's tensor parameters.
    param_shapes: HashMap<String, Vec<i64>>,
    steps: Vec<Step>,
}

pub fn plan(program: &Program, name: &str, shapes: &HashMap<String, i64>) -> Result<Plan, String> {
    let f = program.functions.iter().find(|f| f.name == name).ok_or_else(|| format!("no function `{name}`"))?;
    for p in &f.shape_params {
        if !shapes.contains_key(p) {
            return Err(format!("shape parameter `{p}` of `{name}` is not bound"));
        }
    }
    let env: HashMap<String, i64> = shapes.clone();
    let mut param_shapes = HashMap::new();
    for (pname, ty) in &f.params {
        if let Some(s) = ty.shaped() {
            let dims = s.shape.iter().map(|d| eval(d, &env)).collect::<Result<Vec<_>, _>>()?;
            param_shapes.insert(pname.clone(), dims);
        }
    }
    let mut p = Planner { program, f, env, param_shapes, steps: Vec::new() };
    p.block(&f.body)?;
    Ok(Plan { function: name.to_string(), steps: p.steps })
}

fn eval(s: &Sym, env: &HashMap<String, i64>) -> Result<i64, String> {
    s.eval(&|p| env.get(p).copied()).ok_or_else(|| format!("cannot evaluate `{s}` in a composition"))
}

impl<'a> Planner<'a> {
    fn block(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for s in stmts {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match &s.kind {
            StmtKind::Range { var, lo, hi, body } => {
                let VarKind::Index(Atom::Param(atom)) = &self.f.vars[*var].kind else { unreachable!() };
                let (lo, hi) = (eval(lo, &self.env)?, eval(hi, &self.env)?);
                for i in lo..hi {
                    self.env.insert(atom.clone(), i);
                    self.block(body)?;
                }
                self.env.remove(atom);
                Ok(())
            }
            StmtKind::If { cond, then, els } => {
                if self.cond(cond)? {
                    self.block(then)
                } else {
                    self.block(els)
                }
            }
            StmtKind::Expr(e) => match &e.kind {
                ExprKind::Call { callee, shape_args, args, .. } => self.call(callee, shape_args, args),
                _ => Err(format!("at byte {}: a composition statement must be a kernel call", e.span.start)),
            },
            other => Err(format!("at byte {}: `{}` is not allowed in a composition; only kernel calls, `range` and `if`", s.span.start, stmt_name(other))),
        }
    }

    fn cond(&self, e: &Expr) -> Result<bool, String> {
        use crate::ast::BinaryOp;
        match &e.kind {
            ExprKind::Bool(b) => Ok(*b),
            ExprKind::Binary { op, lhs, rhs } => match op {
                BinaryOp::And => Ok(self.cond(lhs)? && self.cond(rhs)?),
                BinaryOp::Or => Ok(self.cond(lhs)? || self.cond(rhs)?),
                _ => {
                    let (Some(l), Some(r)) = (&lhs.sym, &rhs.sym) else {
                        return Err(format!("at byte {}: a composition condition must compare static integers", e.span.start));
                    };
                    let (l, r) = (eval(l, &self.env)?, eval(r, &self.env)?);
                    Ok(match op {
                        BinaryOp::Eq => l == r,
                        BinaryOp::Ne => l != r,
                        BinaryOp::Lt => l < r,
                        BinaryOp::Le => l <= r,
                        BinaryOp::Gt => l > r,
                        BinaryOp::Ge => l >= r,
                        _ => return Err(format!("at byte {}: unsupported condition in a composition", e.span.start)),
                    })
                }
            },
            ExprKind::Unary { op: crate::ast::UnaryOp::Not, expr } => Ok(!self.cond(expr)?),
            _ => Err(format!("at byte {}: unsupported condition in a composition", e.span.start)),
        }
    }

    fn call(&mut self, callee: &str, shape_args: &[Sym], args: &[Expr]) -> Result<(), String> {
        let sig = self.program.signatures.get(callee).ok_or_else(|| format!("no function `{callee}`"))?;
        let mut shapes = HashMap::new();
        for (p, s) in sig.shape_params.iter().zip(shape_args) {
            shapes.insert(p.clone(), eval(s, &self.env)?);
        }
        let mut tensors = Vec::new();
        let mut scalars = Vec::new();
        for (a, (pname, pty)) in args.iter().zip(&sig.params) {
            match pty {
                Ty::Tensor(_) => {
                    let (root, elem_offset, elem) = self.view(a)?;
                    tensors.push(TensorBinding { param: pname.clone(), root, elem_offset, elem });
                }
                Ty::Scalar(_) => scalars.push((pname.clone(), self.scalar(a)?)),
                other => return Err(format!("at byte {}: cannot pass a {other} to a kernel from a composition", a.span.start)),
            }
        }
        self.steps.push(Step { kernel: callee.to_string(), shapes, tensors, scalars });
        Ok(())
    }

    /// A contiguous view of a composition parameter: the parameter itself, or points on its
    /// leading axes with the remaining axes whole.
    fn view(&self, e: &Expr) -> Result<(String, i64, Elem), String> {
        match &e.kind {
            ExprKind::Var(v) => {
                let var = &self.f.vars[*v];
                let Ty::Tensor(s) = &var.ty else { return Err(format!("at byte {}: `{}` is not a tensor", e.span.start, var.name)) };
                if !matches!(var.kind, VarKind::Param(_)) {
                    return Err(format!("at byte {}: `{}` is not a parameter", e.span.start, var.name));
                }
                Ok((var.name.clone(), 0, s.elem.clone()))
            }
            ExprKind::Index { base, indices } => {
                let ExprKind::Var(v) = base.kind else { return Err(format!("at byte {}: a kernel argument must index a parameter directly", e.span.start)) };
                let var = &self.f.vars[v];
                let Ty::Tensor(s) = &var.ty else { return Err(format!("at byte {}: `{}` is not a tensor", e.span.start, var.name)) };
                let dims = &self.param_shapes[&var.name];
                // Contiguity: every axis before the last partial axis selects one element
                // (a point or a slice of extent 1); every axis after it is whole.
                let mut offset = 0i64;
                let mut widened = false;
                for (axis, idx) in indices.iter().enumerate() {
                    let stride: i64 = dims[axis + 1..].iter().product();
                    let static_int = |x: &Expr| -> Result<i64, String> {
                        let sym = x.sym.as_ref().ok_or_else(|| format!("at byte {}: a kernel argument index must be static", x.span.start))?;
                        eval(sym, &self.env)
                    };
                    let (lo, hi) = match idx {
                        Index::Point(p) => {
                            let i = static_int(p)?;
                            (i, i + 1)
                        }
                        Index::Slice { start, end } => {
                            let lo = match start {
                                Some(x) => static_int(x)?,
                                None => 0,
                            };
                            let hi = match end {
                                Some(x) => static_int(x)?,
                                None => dims[axis],
                            };
                            (lo, hi)
                        }
                    };
                    if lo < 0 || hi > dims[axis] || lo >= hi {
                        return Err(format!("at byte {}: range {lo}..{hi} is out of range for extent {}", e.span.start, dims[axis]));
                    }
                    let whole = lo == 0 && hi == dims[axis];
                    if widened && !whole {
                        return Err(format!("at byte {}: a kernel argument view must be contiguous: after a partial axis every axis must be whole", e.span.start));
                    }
                    if hi - lo > 1 && !whole {
                        widened = true;
                    }
                    if whole && hi - lo > 1 {
                        widened = true;
                    }
                    offset += lo * stride;
                }
                Ok((var.name.clone(), offset, s.elem.clone()))
            }
            _ => Err(format!("at byte {}: a kernel argument must be a parameter or a view of one", e.span.start)),
        }
    }

    fn scalar(&self, e: &Expr) -> Result<ScalarSource, String> {
        match &e.kind {
            ExprKind::Float(x) => Ok(ScalarSource::Literal(*x)),
            ExprKind::Int(i) => Ok(ScalarSource::Literal(*i as f64)),
            ExprKind::Var(v) => {
                let var = &self.f.vars[*v];
                let is_param = matches!(var.kind, VarKind::Param(_)) || matches!(&var.kind, VarKind::Index(Atom::Param(p)) if self.f.params.iter().any(|(n, _)| n == p));
                if !is_param {
                    return Err(format!("at byte {}: `{}` is not a scalar parameter of the composition", e.span.start, var.name));
                }
                Ok(ScalarSource::Param(var.name.clone()))
            }
            ExprKind::Cast { .. } | ExprKind::Unary { .. } | ExprKind::Binary { .. } => {
                match &e.sym {
                    Some(s) => Ok(ScalarSource::Literal(eval(s, &self.env)? as f64)),
                    None => Err(format!("at byte {}: a scalar argument must be a literal or a parameter", e.span.start)),
                }
            }
            _ => Err(format!("at byte {}: a scalar argument must be a literal or a parameter", e.span.start)),
        }
    }
}

fn stmt_name(s: &StmtKind) -> &'static str {
    match s {
        StmtKind::Parallel { .. } => "parallel",
        StmtKind::LoadLoop { .. } => "load",
        StmtKind::Owned { .. } => "owned",
        StmtKind::Range { .. } => "range",
        StmtKind::Lanes { .. } => "lanes",
        StmtKind::If { .. } => "if",
        StmtKind::Assign { .. } => "assignment",
        StmtKind::Expr(_) => "expression",
    }
}
