//! The checker: name resolution by scope, typing, symbolic values, bounds,
//! definite assignment, parallel independence, and lowering residuals.

use crate::ast;
use crate::ast::{AssignOp, BinaryOp, ExprKind as A, Ident, UnaryOp};
use crate::hir::*;
use crate::intrinsics::{self, Intrinsic, IntrinsicParam, IntrinsicResult};
use crate::repr;
use crate::span::{Diagnostic, Span};
use crate::sym::{Atom, Facts, Prover, Sym};
use crate::types::{DType, Elem, Shaped, Ty};
use crate::Scope;
use std::collections::{HashMap, HashSet};

/// A declaration's interface as seen by callers.
#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    pub name: String,
    pub is_construct: bool,
    pub shape_params: Vec<String>,
    pub elem_params: Vec<String>,
    pub params: Vec<(String, Ty)>,
    /// Scalar parameters declared `index[E]`: an i32 the caller guarantees in [0, E).
    pub index_params: Vec<(String, Sym)>,
    pub span: Span,
}

/// What the checker needs from the program: every signature, and the file's scope.
pub struct Env<'a> {
    pub signatures: &'a HashMap<String, Signature>,
    pub scope: Scope,
}

pub struct Checked {
    pub functions: Vec<Function>,
    pub lowerings: Vec<Lowering>,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn check_file(file: &ast::File, env: &Env) -> Checked {
    let mut out = Checked { functions: Vec::new(), lowerings: Vec::new(), diagnostics: Vec::new() };
    for decl in &file.decls {
        match decl {
            ast::Decl::Fn(f) | ast::Decl::Construct(f) => {
                let is_construct = matches!(decl, ast::Decl::Construct(_));
                if env.scope != Scope::Portable && is_construct {
                    out.diagnostics.push(Diagnostic::new(f.name.span, "`construct` is only allowed in portable files"));
                    continue;
                }
                let sig = match signature_of(f, is_construct) {
                    Ok(s) => s,
                    Err(d) => {
                        out.diagnostics.push(d);
                        continue;
                    }
                };
                let mut c = Checker::new(env, &sig, env.scope.clone(), false);
                let body = c.block(&f.body, &[]);
                out.diagnostics.append(&mut c.diagnostics);
                out.functions.push(Function {
                    name: sig.name.clone(),
                    is_construct,
                    shape_params: sig.shape_params.clone(),
                    elem_params: sig.elem_params.clone(),
                    params: sig.params.clone(),
                    vars: c.vars,
                    body,
                });
            }
            ast::Decl::Lower(l) => {
                let backend = match &env.scope {
                    Scope::Backend(b) => b.clone(),
                    Scope::Portable => {
                        out.diagnostics.push(Diagnostic::new(l.name.span, "`lower` is only allowed in backend files"));
                        continue;
                    }
                };
                let Some(sig) = env.signatures.get(&l.name.name) else {
                    out.diagnostics.push(Diagnostic::new(l.name.span, format!("`{}` is not declared", l.name.name)));
                    continue;
                };
                if !sig.is_construct {
                    out.diagnostics.push(Diagnostic::new(l.name.span, format!("`{}` is a kernel (`fn`); only constructs have lowerings", l.name.name)));
                    continue;
                }
                let Some(body) = &l.body else {
                    out.lowerings.push(Lowering { construct: sig.name.clone(), backend, elem_bindings: Vec::new(), vars: Vec::new(), body: Vec::new(), residual: Vec::new() });
                    continue;
                };
                // The block restates the construct's signature. It must be a valid
                // specialization: the same shape parameters and parameter list, with a
                // concrete element type permitted where the construct has an element
                // parameter. Those substitutions are the block's specialization, read from
                // the signature rather than declared beside it.
                let (specialized, elem_bindings) = match specialize(sig, l, &mut out.diagnostics) {
                    Some(x) => x,
                    None => continue,
                };
                let mut c = Checker::new(env, &specialized, Scope::Backend(backend.clone()), true);
                let stmts = c.block(body, &[]);
                c.output_coverage(&stmts);
                out.diagnostics.append(&mut c.diagnostics);
                out.lowerings.push(Lowering { construct: sig.name.clone(), backend, elem_bindings, vars: c.vars, body: stmts, residual: c.residual });
            }
        }
    }
    out
}

/// Build a declaration's signature. Shapes are symbolic over its own shape parameters.
pub fn signature_of(f: &ast::FnDecl, is_construct: bool) -> Result<Signature, Diagnostic> {
    let shape_params: Vec<String> = f.shape.iter().map(|s| s.name.clone()).collect();
    let mut elem_params = Vec::new();
    let mut params = Vec::new();
    let mut index_params = Vec::new();
    let mut seen = HashSet::new();
    for p in &f.params {
        if !seen.insert(p.name.name.clone()) {
            return Err(Diagnostic::new(p.name.span, format!("duplicate parameter `{}`", p.name.name)));
        }
        if p.ty.head.name == "index" {
            if p.ty.shape.len() != 1 || p.ty.elem.is_some() {
                return Err(Diagnostic::new(p.ty.span, "`index[E]` takes one extent and no element type"));
            }
            let bound = shape_sym(&p.ty.shape[0], &shape_params)?;
            index_params.push((p.name.name.clone(), bound));
            params.push((p.name.name.clone(), Ty::Scalar(DType::I32)));
            continue;
        }
        let ty = type_from_ast(&p.ty, &shape_params, &mut elem_params)?;
        params.push((p.name.name.clone(), ty));
    }
    Ok(Signature { name: f.name.name.clone(), is_construct, shape_params, elem_params, params, index_params, span: f.name.span })
}

fn type_from_ast(t: &ast::TypeExpr, shape_params: &[String], elem_params: &mut Vec<String>) -> Result<Ty, Diagnostic> {
    let elem_of = |name: &Ident, elem_params: &mut Vec<String>| -> Result<Elem, Diagnostic> {
        if let Some(d) = DType::from_name(&name.name) {
            return Ok(Elem::Dtype(d));
        }
        if repr::lookup(&name.name).is_some() {
            return Ok(Elem::Repr(name.name.clone()));
        }
        if name.name.len() <= 2 && name.name.chars().all(|c| c.is_ascii_uppercase()) {
            if !elem_params.contains(&name.name) {
                elem_params.push(name.name.clone());
            }
            return Ok(Elem::Param(name.name.clone()));
        }
        Err(Diagnostic::new(name.span, format!("`{}` is not a dtype, a representation, or a dtype parameter", name.name)))
    };
    match t.head.name.as_str() {
        "tensor" | "tile" => {
            let Some(elem) = &t.elem else {
                return Err(Diagnostic::new(t.span, format!("`{}` needs an element type", t.head.name)));
            };
            if t.shape.is_empty() {
                return Err(Diagnostic::new(t.span, format!("`{}` needs a shape", t.head.name)));
            }
            let mut shape = Vec::new();
            for e in &t.shape {
                shape.push(shape_sym(e, shape_params)?);
            }
            let s = Shaped::new(shape, elem_of(elem, elem_params)?);
            Ok(if t.head.name == "tensor" { Ty::Tensor(s) } else { Ty::Tile(s) })
        }
        other => {
            if !t.shape.is_empty() || t.elem.is_some() {
                return Err(Diagnostic::new(t.span, format!("`{other}` does not take a shape or element type")));
            }
            match DType::from_name(other) {
                Some(d) => Ok(Ty::Scalar(d)),
                None => Err(Diagnostic::new(t.head.span, format!("unknown type `{other}`"))),
            }
        }
    }
}

/// A shape expression in a signature: integers, shape parameters, and arithmetic over them.
fn shape_sym(e: &ast::Expr, shape_params: &[String]) -> Result<Sym, Diagnostic> {
    match &e.kind {
        A::Int(v) => Ok(Sym::constant(*v as i64)),
        A::Name(n) if shape_params.contains(&n.name) => Ok(Sym::param(&n.name)),
        A::Name(n) => Err(Diagnostic::new(n.span, format!("`{}` is not a declared shape parameter", n.name))),
        A::Binary { op, lhs, rhs } => {
            let l = shape_sym(lhs, shape_params)?;
            let r = shape_sym(rhs, shape_params)?;
            match op {
                BinaryOp::Add => Ok(l.add(&r)),
                BinaryOp::Sub => Ok(l.sub(&r)),
                BinaryOp::Mul => Ok(l.mul(&r)),
                BinaryOp::Div => Ok(l.quot(&r)),
                BinaryOp::Rem => Ok(l.rem(&r)),
                _ => Err(Diagnostic::new(e.span, "only + - * / % are allowed in shapes")),
            }
        }
        _ => Err(Diagnostic::new(e.span, "shape must be an integer expression over shape parameters")),
    }
}

// ---------------------------------------------------------------------------

struct Checker<'a> {
    env: &'a Env<'a>,
    sig: &'a Signature,
    scope: Scope,
    intrinsics: Vec<Intrinsic>,
    is_lowering: bool,
    vars: Vec<Var>,
    scopes: Vec<HashMap<String, VarId>>,
    facts: Facts,
    /// Tiles allocated with `tile[...]` and not yet fully assigned.
    unassigned: HashSet<VarId>,
    /// Innermost enclosing `parallel` loop's index variables.
    parallel_vars: Vec<Vec<VarId>>,
    /// Tiles whose first element-wise write inside the current `owned` loop assigns them:
    /// the loop's own tile and any unassigned tile of the same shape written at the loop indices.
    pending_full_assign: Vec<VarId>,
    /// One atom per dynamic slice, keyed by the slice's source span, so a slice applied to a
    /// tuple of views yields the same extent for every element.
    dyn_slices: HashMap<(u32, u32), Atom>,
    diagnostics: Vec<Diagnostic>,
    residual: Vec<Sym>,
    counter: usize,
}

impl<'a> Checker<'a> {
    fn new(env: &'a Env<'a>, sig: &'a Signature, scope: Scope, is_lowering: bool) -> Checker<'a> {
        let intrinsics = match &scope {
            Scope::Backend(b) => intrinsics::table(b).unwrap_or_default(),
            Scope::Portable => Vec::new(),
        };
        let mut c = Checker {
            env,
            sig,
            scope,
            intrinsics,
            is_lowering,
            vars: Vec::new(),
            scopes: vec![HashMap::new()],
            facts: Facts::new(),
            unassigned: HashSet::new(),
            parallel_vars: Vec::new(),
            pending_full_assign: Vec::new(),
            dyn_slices: HashMap::new(),
            diagnostics: Vec::new(),
            residual: Vec::new(),
            counter: 0,
        };
        for p in &sig.shape_params {
            // A shape parameter is an extent: at least 1. Empty work is a runtime slice, not a shape.
            c.facts.set_range_lower(Atom::Param(p.clone()), Sym::constant(1));
        }
        for (i, (name, ty)) in sig.params.iter().enumerate() {
            let id = c.vars.len();
            let kind = match sig.index_params.iter().find(|(n, _)| n == name) {
                Some((_, bound)) => {
                    // An index parameter is a symbol with a known range, like a loop index.
                    let atom = Atom::Param(name.clone());
                    c.facts.set_range(atom.clone(), Sym::constant(0), bound.sub(&Sym::constant(1)));
                    VarKind::Index(atom)
                }
                None => VarKind::Param(i),
            };
            c.vars.push(Var { name: name.clone(), ty: ty.clone(), span: sig.span, kind });
            c.scopes[0].insert(name.clone(), id);
        }
        c
    }

    fn error(&mut self, span: Span, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic::new(span, message));
    }

    fn lookup(&self, name: &str) -> Option<VarId> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    fn declare(&mut self, name: &str, ty: Ty, span: Span, kind: VarKind) -> VarId {
        let id = self.vars.len();
        self.vars.push(Var { name: name.to_string(), ty, span, kind });
        self.scopes.last_mut().unwrap().insert(name.to_string(), id);
        id
    }

    fn fresh_atom(&mut self, base: &str) -> Atom {
        self.counter += 1;
        Atom::Param(format!("{base}#{}", self.counter))
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn prover(&self) -> Prover<'_> {
        Prover::new(&self.facts)
    }

    /// Prove `e >= 0`; on failure report an error, or in a lowering record a residual.
    fn require_nonneg(&mut self, e: &Sym, span: Span, what: &str) {
        if self.prover().nonneg(e) {
            return;
        }
        if self.is_lowering && e.params().iter().all(|p| self.sig.shape_params.contains(p) || p.contains('#')) {
            // Eliminate the loop atoms by their ranges; shape parameters stay exact so the
            // residual is the precise condition under which the body is well-formed.
            let iv = self.prover().interval_over(e, &|a| a.mentions_loop());
            if iv.lo.params().iter().all(|p| self.sig.shape_params.contains(p)) {
                if !self.residual.contains(&iv.lo) {
                    self.residual.push(iv.lo);
                }
                return;
            }
        }
        self.error(span, format!("{what}: cannot prove `{e} >= 0`"));
    }

    // ---- statements ----

    fn block(&mut self, b: &ast::Block, _outer: &[VarId]) -> Vec<Stmt> {
        let mut out = Vec::new();
        for s in &b.stmts {
            if let Some(st) = self.stmt(s) {
                out.push(st);
            }
        }
        out
    }

    fn stmt(&mut self, s: &ast::Stmt) -> Option<Stmt> {
        let kind = match &s.kind {
            ast::StmtKind::For { targets, iter, body } => self.for_stmt(targets, iter, body, s.span)?,
            ast::StmtKind::If { cond, then, els } => {
                let cond = self.expr(cond, Some(&Ty::Scalar(DType::Bool)))?;
                if cond.ty != Ty::Scalar(DType::Bool) {
                    self.error(cond.span, format!("condition must be bool, found {}", cond.ty));
                }
                // Path facts: a comparison on symbolic integers bounds its atoms inside the
                // branch it guards, and its negation bounds them in the other branch.
                let mark = self.facts.path_mark();
                for (atom, lo, hi) in path_bounds(&cond, false) {
                    if let Some(lo) = lo {
                        self.facts.add_lower(atom.clone(), lo);
                    }
                    if let Some(hi) = hi {
                        self.facts.add_upper(atom, hi);
                    }
                }
                self.push_scope();
                let then = self.block(then, &[]);
                self.pop_scope();
                self.facts.path_rollback(mark);
                let els = match els {
                    Some(b) => {
                        for (atom, lo, hi) in path_bounds(&cond, true) {
                            if let Some(lo) = lo {
                                self.facts.add_lower(atom.clone(), lo);
                            }
                            if let Some(hi) = hi {
                                self.facts.add_upper(atom, hi);
                            }
                        }
                        self.push_scope();
                        let e = self.block(b, &[]);
                        self.pop_scope();
                        e
                    }
                    None => Vec::new(),
                };
                self.facts.path_rollback(mark);
                StmtKind::If { cond, then, els }
            }
            ast::StmtKind::Assign { target, op, value } => self.assign(target, *op, value)?,
            ast::StmtKind::Expr(e) => {
                let e = self.expr(e, None)?;
                if e.ty != Ty::Void {
                    self.error(e.span, "expression statement must be a call that returns nothing");
                }
                StmtKind::Expr(e)
            }
        };
        Some(Stmt { kind, span: s.span })
    }

    fn for_stmt(&mut self, targets: &[Ident], iter: &ast::Expr, body: &ast::Block, _span: Span) -> Option<StmtKind> {
        // `parallel` is a bare word: it names indices and takes no arguments at all.
        let no_args = Vec::new();
        let (callee_name, args) = match &iter.kind {
            A::Name(n) if n.name == "parallel" => (n, &no_args),
            A::Call { callee, args, .. } => match &callee.kind {
                A::Name(n) => (n, args),
                _ => {
                    self.error(callee.span, "unknown iterator");
                    return None;
                }
            },
            _ => {
                self.error(iter.span, "`for` iterates `parallel`, `load(..., over=axis)`, `owned(tile)` or `range(...)`");
                return None;
            }
        };
        match callee_name.name.as_str() {
            "parallel" => {
                if self.is_lowering {
                    self.error(iter.span, "`parallel` is not allowed in a lowering; the construct's work items are fixed by the caller");
                }
                // `parallel` names its indices and nothing else. How much work there is
                // follows from the regions the body stores, so a kernel cannot state, and
                // therefore cannot leak, how that work is divided.
                if !args.is_empty() {
                    self.error(iter.span, "`parallel` takes only index names; the number of work items follows from what the body stores");
                    return None;
                }
                self.push_scope();
                let mut vars = Vec::new();
                for t in targets {
                    let atom = self.fresh_atom(&t.name);
                    vars.push(self.declare(&t.name, Ty::Scalar(DType::I32), t.span, VarKind::Index(atom)));
                }
                // The body is checked twice: once to find the extents from its stores, then
                // again with those extents as facts so index bounds can be proved.
                let saved_errors = self.diagnostics.len();
                let saved_counter = self.counter;
                let probe = self.block(body, &vars);
                let extents = self.derive_extents(&vars, &probe, iter.span);
                self.diagnostics.truncate(saved_errors);
                self.pop_scope();

                self.counter = saved_counter;
                self.push_scope();
                let mut vars = Vec::new();
                for (t, ext) in targets.iter().zip(&extents) {
                    let atom = self.fresh_atom(&t.name);
                    self.facts.set_range(atom.clone(), Sym::constant(0), ext.sub(&Sym::constant(1)));
                    vars.push(self.declare(&t.name, Ty::Scalar(DType::I32), t.span, VarKind::Index(atom)));
                }
                self.parallel_vars.push(vars.clone());
                let body = self.block(body, &vars);
                self.parallel_vars.pop();
                self.pop_scope();
                Some(StmtKind::Parallel { vars, extents, body })
            }
            "owned" => {
                if args.len() != 1 || args[0].name.is_some() {
                    self.error(iter.span, "`owned` takes exactly one tile");
                    return None;
                }
                let tile = self.expr_inner(&args[0].value, None, true)?;
                let Ty::Tile(shaped) = &tile.ty else {
                    self.error(tile.span, format!("`owned` needs a tile, found {}", tile.ty));
                    return None;
                };
                if shaped.shape.len() != targets.len() {
                    self.error(iter.span, format!("tile has rank {} but the loop binds {} names", shaped.shape.len(), targets.len()));
                    return None;
                }
                let shape = shaped.shape.clone();
                self.push_scope();
                let mut vars = Vec::new();
                for (t, ext) in targets.iter().zip(&shape) {
                    let atom = self.fresh_atom(&t.name);
                    self.facts.set_range(atom.clone(), Sym::constant(0), ext.sub(&Sym::constant(1)));
                    vars.push(self.declare(&t.name, Ty::Scalar(DType::I32), t.span, VarKind::Index(atom)));
                }
                // Definite assignment: an owned loop over an unassigned tile that assigns
                // `t[vars] = ...` as its first write makes the tile assigned.
                let mut pending = Vec::new();
                for v in self.unassigned.iter().copied().collect::<Vec<_>>() {
                    let same_shape = matches!(&self.vars[v].ty, Ty::Tile(s) if s.shape == shape);
                    if same_shape && body_assigns_all(body, &self.vars[v].name, targets) {
                        pending.push(v);
                    }
                }
                let saved = std::mem::replace(&mut self.pending_full_assign, pending);
                let stmts = self.block(body, &vars);
                self.pending_full_assign = saved;
                self.pop_scope();
                Some(StmtKind::Owned { vars, tile, body: stmts })
            }
            "range" => {
                if targets.len() != 1 {
                    self.error(iter.span, "`range` binds exactly one name");
                    return None;
                }
                let (lo, hi) = match args.len() {
                    1 => (Sym::constant(0), self.int_arg(&args[0])?),
                    2 => (self.int_arg(&args[0])?, self.int_arg(&args[1])?),
                    _ => {
                        self.error(iter.span, "`range` takes one or two integer expressions");
                        return None;
                    }
                };
                self.push_scope();
                let atom = self.fresh_atom(&targets[0].name);
                self.facts.set_range(atom.clone(), lo.clone(), hi.sub(&Sym::constant(1)));
                let var = self.declare(&targets[0].name, Ty::Scalar(DType::I32), targets[0].span, VarKind::Index(atom));
                let body = self.block(body, &[var]);
                self.pop_scope();
                Some(StmtKind::Range { var, lo, hi, body })
            }
            "lanes" => {
                if self.scope == Scope::Portable {
                    self.error(iter.span, "`lanes` is only allowed in a lowering");
                    return None;
                }
                if targets.len() != 1 || args.is_empty() || args.len() > 2 {
                    self.error(iter.span, "`lanes(extent)` or `lanes(extent, width)` binds exactly one name");
                    return None;
                }
                let hi = self.int_arg(&args[0])?;
                let width = if args.len() == 2 { self.int_arg(&args[1])?.as_constant().unwrap_or(1) } else { 1 };
                // Lanes cover the extent in runs of `width` per lane across the subgroup.
                let run = Sym::constant(32 * width);
                let need = hi.sub(&hi.quot(&run).mul(&run));
                self.require_nonneg(&need.neg(), iter.span, "lanes: extent must be a multiple of the subgroup run");
                self.push_scope();
                let atom = self.fresh_atom(&targets[0].name);
                self.facts.set_range(atom.clone(), Sym::constant(0), hi.sub(&Sym::constant(1)));
                let var = self.declare(&targets[0].name, Ty::Scalar(DType::I32), targets[0].span, VarKind::Index(atom));
                let body = self.block(body, &[var]);
                self.pop_scope();
                Some(StmtKind::Lanes { var, extent: hi, width, body })
            }
            "load" => {
                let mut axis = None;
                let mut views_ast = None;
                for a in args {
                    match &a.name {
                        Some(n) if n.name == "over" => {
                            let e = self.expr(&a.value, Some(&Ty::Scalar(DType::I32)))?;
                            match e.sym.as_ref().and_then(|s| s.as_constant()) {
                                Some(v) if v >= 0 => axis = Some(v as usize),
                                _ => self.error(e.span, "`over` must be a constant axis index"),
                            }
                        }
                        Some(n) => self.error(n.span, format!("unknown argument `{}`", n.name)),
                        None if views_ast.is_none() => views_ast = Some(&a.value),
                        None => self.error(a.value.span, "`load` takes one view or a tuple of views"),
                    }
                }
                let (Some(axis), Some(views_ast)) = (axis, views_ast) else {
                    self.error(iter.span, "a `load` loop needs a view and `over=axis`");
                    return None;
                };
                let views_expr = self.expr(views_ast, None)?;
                let views: Vec<Expr> = match views_expr.kind {
                    ExprKind::Tuple(items) => items,
                    _ => vec![views_expr],
                };
                if views.len() != targets.len() {
                    self.error(iter.span, format!("`load` yields {} tiles but the loop binds {} names", views.len(), targets.len()));
                    return None;
                }
                let mut extent: Option<Sym> = None;
                for v in &views {
                    let Ty::Tensor(s) = &v.ty else {
                        self.error(v.span, format!("`load` needs a tensor view, found {}", v.ty));
                        return None;
                    };
                    let Some(dim) = s.shape.get(axis) else {
                        self.error(v.span, format!("axis {axis} is out of range for rank {}", s.shape.len()));
                        return None;
                    };
                    match &extent {
                        None => extent = Some(dim.clone()),
                        Some(e) if e != dim => self.error(v.span, format!("views streamed together must agree on axis {axis}: `{e}` vs `{dim}`")),
                        _ => {}
                    }
                }
                let extent = extent.unwrap();
                let piece = self.fresh_atom("piece");
                self.facts.set_range(piece.clone(), Sym::constant(1), extent.clone());
                self.push_scope();
                let mut vars = Vec::new();
                for (t, v) in targets.iter().zip(&views) {
                    let Ty::Tensor(s) = &v.ty else { unreachable!() };
                    let mut shape = s.shape.clone();
                    shape[axis] = Sym::atom(piece.clone());
                    let ty = Ty::Tile(Shaped { shape, elem: s.elem.clone(), packed_axis: s.packed_axis });
                    vars.push(self.declare(&t.name, ty, t.span, VarKind::Local));
                }
                let body = self.block(body, &vars);
                self.pop_scope();
                Some(StmtKind::LoadLoop { vars, views, axis, piece, capacity: None, body })
            }
            other => {
                self.error(callee_name.span, format!("`{other}` is not an iterator"));
                None
            }
        }
    }

    fn int_arg(&mut self, a: &ast::Arg) -> Option<Sym> {
        if let Some(n) = &a.name {
            self.error(n.span, "unexpected keyword argument");
        }
        let e = self.expr(&a.value, Some(&Ty::Scalar(DType::I32)))?;
        match e.sym {
            Some(s) => Some(s),
            None => {
                self.error(e.span, "expected a static integer expression");
                None
            }
        }
    }

    fn assign(&mut self, target: &ast::Expr, op: AssignOp, value: &ast::Expr) -> Option<StmtKind> {
        // Declaration or reassignment of a whole variable.
        if let A::Name(n) = &target.kind {
            if op != AssignOp::Assign {
                let Some(id) = self.lookup(&n.name) else {
                    self.error(n.span, format!("`{}` is not declared", n.name));
                    return None;
                };
                let ty = self.vars[id].ty.clone();
                let value = self.expr(value, Some(&ty))?;
                self.check_arith_assign(&ty, &value, op)?;
                let target = Expr { kind: ExprKind::Var(id), ty, sym: None, span: n.span };
                return Some(StmtKind::Assign { target, op, value });
            }
            let existing = self.lookup(&n.name);
            let expected = existing.map(|id| self.vars[id].ty.clone());
            let value = self.expr(value, expected.as_ref())?;
            if value.ty == Ty::Void {
                self.error(value.span, "cannot assign a call that returns nothing");
                return None;
            }
            let id = match existing {
                Some(id) => {
                    let ty = self.vars[id].ty.clone();
                    if matches!(self.vars[id].kind, VarKind::Index(_)) {
                        self.error(n.span, format!("`{}` is a loop index and cannot be assigned", n.name));
                        return None;
                    }
                    if !self.assignable(&ty, &value.ty) {
                        self.error(value.span, format!("`{}` has type {} but the value has type {}", n.name, ty, value.ty));
                        return None;
                    }
                    self.unassigned.remove(&id);
                    id
                }
                None => {
                    let ty = value.ty.clone();
                    // A scalar argmax is an index into the reduced axis: it is a symbol with
                    // that range, so it may serve as a point index.
                    let argmax_range = match (&value.kind, &ty) {
                        (ExprKind::Builtin { name: Builtin::Reduce, args }, Ty::Scalar(DType::I32)) => {
                            let (ExprKind::Int(axis), ExprKind::Int(op)) = (&args[1].kind, &args[2].kind) else { unreachable!() };
                            let shaped = args[0].ty.shaped().unwrap();
                            (*op == ReduceOp::Argmax as i64).then(|| shaped.shape[*axis as usize].clone())
                        }
                        _ => None,
                    };
                    let kind = match argmax_range {
                        Some(extent) => {
                            let atom = self.fresh_atom(&n.name);
                            self.facts.set_range(atom.clone(), Sym::constant(0), extent.sub(&Sym::constant(1)));
                            VarKind::Index(atom)
                        }
                        None => VarKind::Local,
                    };
                    let id = self.declare(&n.name, ty, n.span, kind);
                    if matches!(value.kind, ExprKind::TileAlloc { .. }) {
                        self.unassigned.insert(id);
                    }
                    id
                }
            };
            let ty = self.vars[id].ty.clone();
            let target = Expr { kind: ExprKind::Var(id), ty, sym: None, span: n.span };
            return Some(StmtKind::Assign { target, op, value });
        }
        // Element assignment: `t[i, j] op= value`.
        let A::Index { base, .. } = &target.kind else {
            self.error(target.span, "assignment target must be a name or an indexed tile");
            return None;
        };
        let A::Name(base_name) = &base.kind else {
            self.error(base.span, "element assignment must index a tile variable directly");
            return None;
        };
        let Some(id) = self.lookup(&base_name.name) else {
            self.error(base_name.span, format!("`{}` is not declared", base_name.name));
            return None;
        };
        if !matches!(self.vars[id].ty, Ty::Tile(_)) {
            self.error(target.span, format!("only tile elements can be assigned; `{}` is a {}. Tensors are written with `store` or `atomic`", base_name.name, self.vars[id].ty));
            return None;
        }
        // A full assignment through the pending owned loop marks the tile assigned from here on.
        let is_pending = self.pending_full_assign.contains(&id) && op == AssignOp::Assign;
        if self.unassigned.contains(&id) && !is_pending {
            self.error(target.span, format!("`{}` is written element-wise before it is assigned; assign every element through `for ... in owned({})`", base_name.name, base_name.name));
            return None;
        }
        let allow_unassigned_target = is_pending;
        let t = self.expr_inner(target, None, allow_unassigned_target)?;
        let Ty::Scalar(dtype) = t.ty else {
            self.error(target.span, "assignment target must select a single element");
            return None;
        };
        let value = self.expr(value, Some(&Ty::Scalar(dtype)))?;
        if is_pending {
            self.unassigned.remove(&id);
            self.pending_full_assign.retain(|v| *v != id);
        } else if self.unassigned.contains(&id) {
            self.error(target.span, format!("`{}` is written element-wise before it is assigned; assign every element through `for ... in owned({})`", base_name.name, base_name.name));
            return None;
        }
        let Ty::Scalar(vd) = value.ty else {
            self.error(value.span, format!("cannot assign {} to an element of dtype {}", value.ty, dtype.name()));
            return None;
        };
        if op == AssignOp::Assign {
            if !(vd == dtype || (vd.is_float() && dtype.is_float()) || (vd.is_int() && dtype.is_int() && vd == dtype)) {
                self.error(value.span, format!("cannot assign {} to an element of dtype {}; cast explicitly", vd.name(), dtype.name()));
            }
        } else {
            self.check_arith_assign(&Ty::Scalar(dtype), &value, op)?;
        }
        Some(StmtKind::Assign { target: t, op, value })
    }

    fn check_arith_assign(&mut self, target: &Ty, value: &Expr, op: AssignOp) -> Option<()> {
        let (Ty::Scalar(t), Ty::Scalar(v)) = (target, &value.ty) else {
            if let (Ty::Tile(a), Ty::Tile(b)) = (target, &value.ty) {
                if a.shape != b.shape {
                    self.error(value.span, format!("shape mismatch in `{}`: {} vs {}", op.text(), target, value.ty));
                    return None;
                }
                return Some(());
            }
            self.error(value.span, format!("`{}` needs numeric operands, found {} and {}", op.text(), target, value.ty));
            return None;
        };
        if !t.is_numeric() || DType::promote(*t, *v).is_none() || (t.is_float() && DType::promote(*t, *v) != Some(*t) && *t != DType::F32) {
            self.error(value.span, format!("`{}` between {} and {} needs an explicit cast", op.text(), t.name(), v.name()));
            return None;
        }
        Some(())
    }

    fn assignable(&self, target: &Ty, value: &Ty) -> bool {
        match (target, value) {
            (Ty::Scalar(a), Ty::Scalar(b)) => a == b || (a.is_float() && b.is_float()),
            (Ty::Tile(a), Ty::Tile(b)) => a.shape == b.shape,
            _ => target == value,
        }
    }

    // ---- expressions ----

    fn expr(&mut self, e: &ast::Expr, expected: Option<&Ty>) -> Option<Expr> {
        self.expr_inner(e, expected, false)
    }

    fn expr_inner(&mut self, e: &ast::Expr, expected: Option<&Ty>, allow_unassigned: bool) -> Option<Expr> {
        let span = e.span;
        match &e.kind {
            A::Int(v) => {
                let dtype = match expected {
                    Some(Ty::Scalar(d)) if d.is_numeric() => *d,
                    _ => DType::I32,
                };
                let sym = if dtype.is_int() { Some(Sym::constant(*v as i64)) } else { None };
                let kind = if dtype.is_float() { ExprKind::Float(*v as f64) } else { ExprKind::Int(*v as i64) };
                Some(Expr { kind, ty: Ty::Scalar(dtype), sym, span })
            }
            A::Float(v) => {
                let dtype = match expected {
                    Some(Ty::Scalar(d)) if d.is_float() => *d,
                    _ => DType::F32,
                };
                Some(Expr { kind: ExprKind::Float(*v), ty: Ty::Scalar(dtype), sym: None, span })
            }
            A::Inf => {
                let dtype = match expected {
                    Some(Ty::Scalar(d)) if d.is_float() => *d,
                    _ => DType::F32,
                };
                Some(Expr { kind: ExprKind::Float(f64::INFINITY), ty: Ty::Scalar(dtype), sym: None, span })
            }
            A::Bool(b) => Some(Expr { kind: ExprKind::Bool(*b), ty: Ty::Scalar(DType::Bool), sym: None, span }),
            A::Name(n) => {
                if let Some(id) = self.lookup(&n.name) {
                    if self.unassigned.contains(&id) && !allow_unassigned {
                        self.error(span, format!("`{}` is read before it is assigned", n.name));
                        return None;
                    }
                    let ty = self.vars[id].ty.clone();
                    let sym = match &self.vars[id].kind {
                        VarKind::Index(a) => Some(Sym::atom(a.clone())),
                        _ => None,
                    };
                    return Some(Expr { kind: ExprKind::Var(id), ty, sym, span });
                }
                if self.sig.shape_params.contains(&n.name) {
                    return Some(Expr { kind: ExprKind::ShapeParam(n.name.clone()), ty: Ty::Scalar(DType::I32), sym: Some(Sym::param(&n.name)), span });
                }
                self.error(span, format!("`{}` is not declared", n.name));
                None
            }
            A::Tuple(items) => {
                let mut out = Vec::new();
                for i in items {
                    out.push(self.expr(i, None)?);
                }
                let ty = Ty::Tuple(out.iter().map(|e| e.ty.clone()).collect());
                Some(Expr { kind: ExprKind::Tuple(out), ty, sym: None, span })
            }
            A::Tile { shape, dtype } => {
                let mut dims = Vec::new();
                for s in shape {
                    let d = self.expr(s, Some(&Ty::Scalar(DType::I32)))?;
                    match d.sym {
                        Some(s) => dims.push(s),
                        None => {
                            self.error(d.span, "tile extents must be static integer expressions");
                            return None;
                        }
                    }
                }
                let Some(dt) = DType::from_name(&dtype.name) else {
                    self.error(dtype.span, format!("`{}` is not a dtype", dtype.name));
                    return None;
                };
                let ty = Ty::Tile(Shaped::new(dims.clone(), Elem::Dtype(dt)));
                Some(Expr { kind: ExprKind::TileAlloc { shape: dims, dtype: dt }, ty, sym: None, span })
            }
            A::Index { base, indices } => {
                let base = self.expr_inner(base, None, allow_unassigned)?;
                self.index(base, indices, span)
            }
            A::Attr { base, name } => {
                let base = self.expr(base, None)?;
                self.attr(base, name, span)
            }
            A::Call { callee, bindings, args } => self.call(callee, bindings, args, expected, span),
            A::Unary { op, expr } => {
                let inner = self.expr(expr, expected)?;
                let Ty::Scalar(d) = inner.ty else {
                    self.error(span, format!("unary `{}` needs a scalar, found {}", op.text().trim(), inner.ty));
                    return None;
                };
                let ok = match op {
                    UnaryOp::Neg => d.is_numeric(),
                    UnaryOp::Not => d == DType::Bool,
                    UnaryOp::BitNot => d.is_int(),
                };
                if !ok {
                    self.error(span, format!("unary `{}` is not defined on {}", op.text().trim(), d.name()));
                    return None;
                }
                let sym = match (op, &inner.sym) {
                    (UnaryOp::Neg, Some(s)) => Some(s.neg()),
                    _ => None,
                };
                Some(Expr { kind: ExprKind::Unary { op: *op, expr: Box::new(inner) }, ty: Ty::Scalar(d), sym, span })
            }
            A::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs, expected, span),
            A::Lambda { .. } => {
                self.error(span, "lambdas are not part of the language");
                None
            }
        }
    }

    fn binary(&mut self, op: BinaryOp, lhs: &ast::Expr, rhs: &ast::Expr, expected: Option<&Ty>, span: Span) -> Option<Expr> {
        let is_cmp = matches!(op, BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge);
        let is_logic = matches!(op, BinaryOp::And | BinaryOp::Or);
        let hint = if is_cmp || is_logic { None } else { expected };
        // Check both sides; literals adapt to the other side's dtype.
        let l0 = self.expr(lhs, hint)?;
        let r_hint = match (&l0.ty, hint) {
            (Ty::Scalar(d), _) if !is_logic => Some(Ty::Scalar(*d)),
            _ => hint.cloned(),
        };
        let r0 = self.expr(rhs, r_hint.as_ref())?;
        let l = if matches!(lhs.kind, A::Int(_) | A::Float(_)) {
            if let Ty::Scalar(d) = r0.ty {
                self.expr(lhs, Some(&Ty::Scalar(d)))?
            } else {
                l0
            }
        } else {
            l0
        };
        let r = r0;
        if is_logic {
            if l.ty != Ty::Scalar(DType::Bool) || r.ty != Ty::Scalar(DType::Bool) {
                self.error(span, format!("`{}` needs bool operands, found {} and {}", op.text(), l.ty, r.ty));
                return None;
            }
            return Some(Expr { kind: ExprKind::Binary { op, lhs: Box::new(l), rhs: Box::new(r) }, ty: Ty::Scalar(DType::Bool), sym: None, span });
        }
        match (&l.ty, &r.ty) {
            (Ty::Scalar(a), Ty::Scalar(b)) if matches!(op, BinaryOp::Shl | BinaryOp::Shr) => {
                // A shift amount is any integer; the result has the shifted operand's type.
                if !a.is_int() || !b.is_int() {
                    self.error(span, format!("`{}` needs integer operands, found {} and {}", op.text(), a.name(), b.name()));
                    return None;
                }
                let sym = match (&l.sym, &r.sym, op) {
                    (Some(x), Some(y), BinaryOp::Shl) => y.as_constant().map(|c| x.scale(1 << c)),
                    (Some(x), Some(y), BinaryOp::Shr) => y.as_constant().map(|c| x.quot(&Sym::constant(1 << c))),
                    _ => None,
                };
                Some(Expr { kind: ExprKind::Binary { op, lhs: Box::new(l.clone()), rhs: Box::new(r) }, ty: Ty::Scalar(*a), sym, span })
            }
            (Ty::Scalar(a), Ty::Scalar(b)) => {
                let Some(d) = DType::promote(*a, *b) else {
                    self.error(span, format!("`{}` between {} and {} needs an explicit cast", op.text(), a.name(), b.name()));
                    return None;
                };
                let bit = matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr);
                if bit && !d.is_int() {
                    self.error(span, format!("`{}` needs integer operands, found {}", op.text(), d.name()));
                    return None;
                }
                if !d.is_numeric() {
                    self.error(span, format!("`{}` is not defined on {}", op.text(), d.name()));
                    return None;
                }
                let sym = if d.is_int() {
                    match (&l.sym, &r.sym) {
                        (Some(x), Some(y)) => match op {
                            BinaryOp::Add => Some(x.add(y)),
                            BinaryOp::Sub => Some(x.sub(y)),
                            BinaryOp::Mul => Some(x.mul(y)),
                            BinaryOp::Div => {
                                if !self.prover().nonneg(&y.sub(&Sym::constant(1))) {
                                    self.error(r.span, format!("divisor `{y}` is not provably positive"));
                                    return None;
                                }
                                Some(x.quot(y))
                            }
                            BinaryOp::Rem => {
                                if !self.prover().nonneg(&y.sub(&Sym::constant(1))) {
                                    self.error(r.span, format!("divisor `{y}` is not provably positive"));
                                    return None;
                                }
                                Some(x.rem(y))
                            }
                            _ => None,
                        },
                        _ => None,
                    }
                } else {
                    None
                };
                let ty = if is_cmp { Ty::Scalar(DType::Bool) } else { Ty::Scalar(d) };
                Some(Expr { kind: ExprKind::Binary { op, lhs: Box::new(l), rhs: Box::new(r) }, ty, sym: if is_cmp { None } else { sym }, span })
            }
            (Ty::Tile(a), Ty::Tile(b)) => {
                if is_cmp {
                    self.error(span, "comparison of tiles is not defined");
                    return None;
                }
                if a.shape != b.shape {
                    self.error(span, format!("shape mismatch: {} vs {}", l.ty, r.ty));
                    return None;
                }
                let elem = promote_elem(&a.elem, &b.elem);
                let ty = Ty::Tile(Shaped::new(a.shape.clone(), Elem::Dtype(elem)));
                Some(Expr { kind: ExprKind::Binary { op, lhs: Box::new(l), rhs: Box::new(r) }, ty, sym: None, span })
            }
            (Ty::Tile(a), Ty::Scalar(s)) | (Ty::Scalar(s), Ty::Tile(a)) => {
                if is_cmp {
                    self.error(span, "comparison of tiles is not defined");
                    return None;
                }
                let elem = promote_elem(&a.elem, &Elem::Dtype(*s));
                let ty = Ty::Tile(Shaped::new(a.shape.clone(), Elem::Dtype(elem)));
                Some(Expr { kind: ExprKind::Binary { op, lhs: Box::new(l), rhs: Box::new(r) }, ty, sym: None, span })
            }
            _ => {
                self.error(span, format!("`{}` is not defined between {} and {}", op.text(), l.ty, r.ty));
                None
            }
        }
    }

    fn index(&mut self, base: Expr, indices: &[ast::Index], span: Span) -> Option<Expr> {
        // Indexing distributes over a tuple: `(k, v)[...]` is `(k[...], v[...])`.
        if let ExprKind::Tuple(items) = base.kind {
            let mut out = Vec::new();
            for item in items {
                out.push(self.index(item, indices, span)?);
            }
            let ty = Ty::Tuple(out.iter().map(|e| e.ty.clone()).collect());
            return Some(Expr { kind: ExprKind::Tuple(out), ty, sym: None, span });
        }
        let (shaped, is_tensor) = match &base.ty {
            Ty::Tensor(s) => (s.clone(), true),
            Ty::Tile(s) => (s.clone(), false),
            other => {
                self.error(span, format!("cannot index a {other}"));
                return None;
            }
        };
        if indices.len() > shaped.shape.len() {
            self.error(span, format!("{} indices for rank {}", indices.len(), shaped.shape.len()));
            return None;
        }
        let mut out_shape = Vec::new();
        let mut out_indices = Vec::new();
        let mut packed_axis = shaped.packed_axis;
        for (axis, idx) in indices.iter().enumerate() {
            let extent = shaped.shape[axis].clone();
            match idx {
                ast::Index::Expr(e) => {
                    match packed_axis {
                        Some(p) if p == axis => packed_axis = None,
                        Some(p) if p > axis => packed_axis = Some(p - 1),
                        _ => {}
                    }
                    let i = self.expr(e, Some(&Ty::Scalar(DType::I32)))?;
                    if i.ty != Ty::Scalar(DType::I32) {
                        self.error(i.span, format!("index must be i32, found {}", i.ty));
                        return None;
                    }
                    match &i.sym {
                        Some(s) => {
                            // The prover sees the loop atoms themselves, so path facts on them apply.
                            self.require_nonneg(s, i.span, "index may be negative");
                            self.require_nonneg(&extent.sub(s).sub(&Sym::constant(1)), i.span, &format!("index may exceed extent `{extent}`"));
                        }
                        None => {
                            self.error(i.span, "point index must be a static integer expression; a data-dependent position needs a slice or an index-typed tensor");
                            return None;
                        }
                    }
                    out_indices.push(Index::Point(i));
                }
                ast::Index::Slice { start, end } => {
                    let s = match start {
                        Some(e) => Some(self.expr(e, Some(&Ty::Scalar(DType::I32)))?),
                        None => None,
                    };
                    let en = match end {
                        Some(e) => Some(self.expr(e, Some(&Ty::Scalar(DType::I32)))?),
                        None => None,
                    };
                    let lo = s.as_ref().and_then(|x| x.sym.clone()).unwrap_or_else(|| Sym::constant(0));
                    let hi = en.as_ref().map(|x| x.sym.clone()).unwrap_or(Some(extent.clone()));
                    let new_extent = match hi {
                        Some(h) if s.as_ref().map(|x| x.sym.is_some()).unwrap_or(true) => {
                            self.require_nonneg(&lo, span, "slice start may be negative");
                            self.require_nonneg(&extent.sub(&h), span, &format!("slice end may exceed extent `{extent}`"));
                            h.sub(&lo)
                        }
                        _ => {
                            // Data-dependent bounds: the view is clamped to the extent at run time.
                            let key = {
                                let a = s.as_ref().map(|x| x.span).or(en.as_ref().map(|x| x.span)).unwrap_or(span);
                                let b = en.as_ref().map(|x| x.span).unwrap_or(a);
                                (a.start, b.end)
                            };
                            let a = match self.dyn_slices.get(&key) {
                                Some(a) => a.clone(),
                                None => {
                                    let a = self.fresh_atom("dyn");
                                    self.facts.set_range(a.clone(), Sym::constant(0), extent.clone());
                                    self.dyn_slices.insert(key, a.clone());
                                    a
                                }
                            };
                            Sym::atom(a)
                        }
                    };
                    out_shape.push(new_extent);
                    out_indices.push(Index::Slice { start: s, end: en });
                }
            }
        }
        for axis in indices.len()..shaped.shape.len() {
            out_shape.push(shaped.shape[axis].clone());
        }
        let ty = if out_shape.is_empty() {
            match shaped.elem.read_dtype() {
                Some(d) => Ty::Scalar(d),
                None => Ty::Scalar(DType::F32),
            }
        } else if is_tensor {
            Ty::Tensor(Shaped { shape: out_shape, elem: shaped.elem.clone(), packed_axis })
        } else {
            Ty::Tile(Shaped { shape: out_shape, elem: shaped.elem.clone(), packed_axis })
        };
        Some(Expr { kind: ExprKind::Index { base: Box::new(base), indices: out_indices }, ty, sym: None, span })
    }

    fn attr(&mut self, base: Expr, name: &Ident, span: Span) -> Option<Expr> {
        match name.name.as_str() {
            "T" => {
                let shaped = match &base.ty {
                    Ty::Tensor(s) | Ty::Tile(s) if s.shape.len() == 2 => s.clone(),
                    other => {
                        self.error(span, format!("`.T` needs a rank-2 tensor or tile, found {other}"));
                        return None;
                    }
                };
                if shaped.packed_axis.is_some() {
                    self.error(span, "a packed tensor or tile cannot be transposed; packets run along its last axis. Contractions consume packed operands as `[N, K]`");
                    return None;
                }
                let t = Shaped { shape: vec![shaped.shape[1].clone(), shaped.shape[0].clone()], elem: shaped.elem, packed_axis: shaped.packed_axis.map(|a| 1 - a) };
                let ty = if matches!(base.ty, Ty::Tensor(_)) { Ty::Tensor(t) } else { Ty::Tile(t) };
                Some(Expr { kind: ExprKind::Transpose(Box::new(base)), ty, sym: None, span })
            }
            "words" | "scale" | "bias" => {
                if self.scope == Scope::Portable {
                    self.error(span, format!("`.{}` exposes the packet structure and is only allowed in a lowering", name.name));
                    return None;
                }
                let Ty::Tile(s) = &base.ty else {
                    self.error(span, format!("`.{}` needs a packed tile, found {}", name.name, base.ty));
                    return None;
                };
                let Elem::Repr(r) = &s.elem else {
                    self.error(span, format!("`.{}` needs a packed tile, found element type {}", name.name, s.elem));
                    return None;
                };
                let rep = repr::lookup(r).unwrap();
                let Some(axis) = s.packed_axis else {
                    self.error(span, "this packed tile has no packet axis left; it is an element");
                    return None;
                };
                let k = s.shape[axis].clone();
                // Accessors keep every other axis and replace the packet axis by the packet extent.
                let (extent, dtype) = match name.name.as_str() {
                    "words" => (rep.words_extent(&k), DType::U32),
                    "scale" => (rep.groups_extent(&k), rep.coefficient),
                    _ => {
                        if !rep.has_bias {
                            self.error(span, format!("`{}` has no bias", rep.name));
                            return None;
                        }
                        (rep.groups_extent(&k), rep.coefficient)
                    }
                };
                let mut shape = s.shape.clone();
                shape[axis] = extent;
                let ty = Ty::Tile(Shaped::new(shape, Elem::Dtype(dtype)));
                Some(Expr { kind: ExprKind::Accessor { base: Box::new(base), name: name.name.clone() }, ty, sym: None, span })
            }
            other => {
                self.error(name.span, format!("unknown attribute `{other}`"));
                None
            }
        }
    }

    fn call(&mut self, callee: &ast::Expr, bindings: &[(Ident, ast::Expr)], args: &[ast::Arg], expected: Option<&Ty>, span: Span) -> Option<Expr> {
        let A::Name(name) = &callee.kind else {
            self.error(callee.span, "only named functions can be called");
            return None;
        };
        let name_str = name.name.as_str();
        // dtype casts
        if let Some(d) = DType::from_name(name_str) {
            if args.len() != 1 || args[0].name.is_some() {
                self.error(span, format!("`{name_str}(x)` takes one argument"));
                return None;
            }
            let inner = self.expr(&args[0].value, None)?;
            let Ty::Scalar(from) = inner.ty else {
                self.error(span, format!("cannot cast {} to {}", inner.ty, d.name()));
                return None;
            };
            if !from.is_numeric() || !d.is_numeric() {
                self.error(span, format!("cannot cast {} to {}", from.name(), d.name()));
                return None;
            }
            let sym = if d.is_int() { inner.sym.clone() } else { None };
            return Some(Expr { kind: ExprKind::Cast { dtype: d, expr: Box::new(inner) }, ty: Ty::Scalar(d), sym, span });
        }
        if let Some(b) = Builtin::from_name(name_str) {
            return self.builtin(b, args, expected, span);
        }
        if matches!(name_str, "parallel" | "owned" | "range" | "lanes") {
            self.error(span, format!("`{name_str}` is an iterator and can only appear in `for`"));
            return None;
        }
        if let Some(intr) = self.intrinsics.iter().find(|i| i.name == name_str).cloned() {
            return self.intrinsic_call(&intr, args, span);
        }
        if intrinsics::table("metal").unwrap().iter().any(|i| i.name == name_str) {
            self.error(span, format!("`{name_str}` is a Metal intrinsic and is not available in this file's scope"));
            return None;
        }
        let Some(sig) = self.env.signatures.get(name_str).cloned() else {
            self.error(name.span, format!("`{name_str}` is not declared"));
            return None;
        };
        if self.is_lowering && sig.name == self.sig.name {
            self.error(span, "a lowering may not call the construct it lowers");
            return None;
        }
        if self.sig.is_construct && !sig.is_construct && !self.is_lowering {
            // Allowed by the language: constructs may compose kernels. Nothing to do.
        }
        self.user_call(&sig, bindings, args, span)
    }

    fn user_call(&mut self, sig: &Signature, bindings: &[(Ident, ast::Expr)], args: &[ast::Arg], span: Span) -> Option<Expr> {
        if args.len() != sig.params.len() {
            self.error(span, format!("`{}` takes {} arguments, {} given", sig.name, sig.params.len(), args.len()));
            return None;
        }
        let mut shape_bind: HashMap<String, Sym> = HashMap::new();
        let mut elem_bind: HashMap<String, Elem> = HashMap::new();
        for (p, v) in bindings {
            if !sig.shape_params.contains(&p.name) {
                self.error(p.span, format!("`{}` has no shape parameter `{}`", sig.name, p.name));
                return None;
            }
            let value = self.expr(v, Some(&Ty::Scalar(DType::I32)))?;
            let Some(sym) = value.sym.clone() else {
                self.error(value.span, "a shape binding must be a static integer expression");
                return None;
            };
            shape_bind.insert(p.name.clone(), sym);
        }
        let mut out_args = Vec::new();
        for (arg, (_pname, pty)) in args.iter().zip(&sig.params) {
            if let Some(n) = &arg.name {
                self.error(n.span, "keyword arguments are not used in calls to functions");
            }
            let expected = match pty {
                Ty::Scalar(d) => Some(Ty::Scalar(*d)),
                _ => None,
            };
            let a = self.expr(&arg.value, expected.as_ref())?;
            out_args.push(a);
        }
        // Shapes: bind single-parameter dimensions, then solve compound dimensions with one
        // unknown, until nothing changes; then every dimension must agree.
        let mut dims: Vec<(Sym, Sym)> = Vec::new();
        for (a, (_, pty)) in out_args.iter().zip(&sig.params) {
            if let (Some(p), Some(x)) = (pty.shaped(), a.ty.shaped()) {
                if p.shape.len() == x.shape.len() {
                    dims.extend(p.shape.iter().cloned().zip(x.shape.iter().cloned()));
                }
            }
        }
        // The callee's shape parameters are renamed to private atoms so they never collide
        // with the caller's parameters of the same name.
        let private = |p: &str| Atom::Param(format!("@{p}"));
        let rename = |s: &Sym| -> Sym {
            let mut e = s.clone();
            for p in &sig.shape_params {
                e = e.subst(&Atom::Param(p.clone()), &Sym::atom(private(p)));
            }
            e
        };
        loop {
            let mut changed = false;
            for (pd, ad) in &dims {
                let mut e = rename(pd);
                for (k, v) in shape_bind.iter() {
                    e = e.subst(&private(k), v);
                }
                let unknown: Vec<String> = e.params().into_iter().filter(|p| p.starts_with('@')).map(|p| p[1..].to_string()).collect();
                if let [u] = unknown.as_slice() {
                    // e = c * u + rest, linear in u: u = (ad - rest) / c when c divides exactly.
                    if let Some((c, rest)) = e.linear_in(&private(u)) {
                        let diff = ad.sub(&rest);
                        let value = if c == 1 {
                            Some(diff)
                        } else if c == -1 {
                            Some(diff.neg())
                        } else {
                            diff.div_exact(c)
                        };
                        if let Some(v) = value {
                            shape_bind.insert(u.clone(), v);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for (a, (pname, pty)) in out_args.iter().zip(&sig.params) {
            if !self.unify(pty, &a.ty, &mut shape_bind, &mut elem_bind, &sig.shape_params) {
                self.error(a.span, format!("argument `{pname}` of `{}` expects {} but was given {}", sig.name, pty, a.ty));
                return None;
            }
        }
        let mut shape_args = Vec::new();
        for p in &sig.shape_params {
            match shape_bind.get(p) {
                Some(s) => shape_args.push(s.clone()),
                None => {
                    self.error(span, format!("shape parameter `{p}` of `{}` is not determined by the arguments", sig.name));
                    return None;
                }
            }
        }
        let mut elem_args = Vec::new();
        for p in &sig.elem_params {
            match elem_bind.get(p) {
                Some(e) => elem_args.push(e.clone()),
                None => {
                    self.error(span, format!("element parameter `{p}` of `{}` is not determined by the arguments", sig.name));
                    return None;
                }
            }
        }
        Some(Expr { kind: ExprKind::Call { callee: sig.name.clone(), shape_args, elem_args, args: out_args }, ty: Ty::Void, sym: None, span })
    }

    /// Match a parameter type against an argument type, binding shape and element parameters.
    fn unify(&mut self, param: &Ty, arg: &Ty, shapes: &mut HashMap<String, Sym>, elems: &mut HashMap<String, Elem>, callee_params: &[String]) -> bool {
        match (param, arg) {
            (Ty::Scalar(a), Ty::Scalar(b)) => a == b || (a.is_float() && b.is_float()),
            (Ty::Tensor(p), Ty::Tensor(a)) | (Ty::Tile(p), Ty::Tile(a)) => {
                if p.shape.len() != a.shape.len() {
                    return false;
                }
                for (pd, ad) in p.shape.iter().zip(&a.shape) {
                    if let Some(name) = single_param(pd) {
                        match shapes.get(&name) {
                            Some(bound) if !self.prover().zero(&bound.sub(ad)) => return false,
                            Some(_) => {}
                            None => {
                                shapes.insert(name, ad.clone());
                            }
                        }
                    } else {
                        // A compound shape must match after substitution of bound parameters;
                        // the callee's parameters are renamed first so caller names cannot collide.
                        let mut e = pd.clone();
                        for cp in callee_params {
                            e = e.subst(&Atom::Param(cp.clone()), &Sym::atom(Atom::Param(format!("@{cp}"))));
                        }
                        for (k, v) in shapes.iter() {
                            e = e.subst(&Atom::Param(format!("@{k}")), v);
                        }
                        if !self.prover().zero(&e.sub(ad)) {
                            return false;
                        }
                    }
                }
                match (&p.elem, &a.elem) {
                    (Elem::Param(name), actual) => match elems.get(name) {
                        Some(bound) => bound == actual,
                        None => {
                            let ok = match actual {
                                Elem::Dtype(d) => d.is_float(),
                                Elem::Repr(_) => true,
                                Elem::Param(_) => true,
                            };
                            if ok {
                                elems.insert(name.clone(), actual.clone());
                            }
                            ok
                        }
                    },
                    (x, y) => x == y,
                }
            }
            _ => false,
        }
    }

    fn builtin(&mut self, b: Builtin, args: &[ast::Arg], expected: Option<&Ty>, span: Span) -> Option<Expr> {
        let positional = |c: &mut Checker, args: &[ast::Arg], n: usize, what: &str| -> Option<()> {
            if args.len() != n || args.iter().any(|a| a.name.is_some()) {
                c.error(span, format!("`{what}` takes {n} positional argument(s)"));
                return None;
            }
            Some(())
        };
        match b {
            Builtin::Load => {
                if args.len() != 1 || args[0].name.is_some() {
                    self.error(span, "`load(view)` takes one view; the streaming form `load(view, over=axis)` is only valid in `for`");
                    return None;
                }
                let v = self.expr(&args[0].value, None)?;
                let ty = match &v.ty {
                    Ty::Tensor(s) => Ty::Tile(s.clone()),
                    Ty::Tuple(items) if items.iter().all(|t| matches!(t, Ty::Tensor(_))) => {
                        Ty::Tuple(items.iter().map(|t| Ty::Tile(t.shaped().unwrap().clone())).collect())
                    }
                    other => {
                        self.error(v.span, format!("`load` needs a tensor view, found {other}"));
                        return None;
                    }
                };
                Some(Expr { kind: ExprKind::Builtin { name: b, args: vec![v] }, ty, sym: None, span })
            }
            Builtin::Store => {
                positional(self, args, 2, "store")?;
                let t = self.expr(&args[0].value, None)?;
                let v = self.expr(&args[1].value, None)?;
                let (Ty::Tile(ts), Ty::Tensor(vs)) = (&t.ty, &v.ty) else {
                    self.error(span, format!("`store(tile, view)` was given {} and {}", t.ty, v.ty));
                    return None;
                };
                if ts.shape != vs.shape {
                    self.error(span, format!("`store` shape mismatch: tile {} into view {}", t.ty, v.ty));
                    return None;
                }
                match (&ts.elem, &vs.elem) {
                    (Elem::Dtype(a), Elem::Dtype(b)) if a == b || (a.is_float() && b.is_float()) => {}
                    (_, Elem::Repr(r)) => {
                        self.error(span, format!("cannot store into a packed `{r}` tensor"));
                        return None;
                    }
                    (a, b) => {
                        self.error(span, format!("cannot store {a} elements into a {b} tensor without a cast"));
                        return None;
                    }
                }
                if self.is_lowering && !self.parallel_vars.is_empty() {
                    unreachable!("lowerings have no parallel loops");
                }
                self.check_independent(&v);
                Some(Expr { kind: ExprKind::Builtin { name: b, args: vec![t, v] }, ty: Ty::Void, sym: None, span })
            }
            Builtin::Atomic => {
                positional(self, args, 3, "atomic")?;
                let op = self.expr(&args[0].value, None);
                let _ = op;
                let A::Name(opname) = &args[0].value.kind else {
                    self.error(args[0].value.span, "`atomic` needs an operation name: add, max or min");
                    return None;
                };
                if !matches!(opname.name.as_str(), "add" | "max" | "min") {
                    self.error(opname.span, "`atomic` operation must be add, max or min");
                    return None;
                }
                let v = self.expr(&args[1].value, None)?;
                let Ty::Tensor(vs) = &v.ty else {
                    self.error(v.span, format!("`atomic` target must be a tensor element view, found {}", v.ty));
                    return None;
                };
                let dtype = vs.elem.read_dtype().unwrap_or(DType::F32);
                let value = self.expr(&args[2].value, Some(&Ty::Scalar(dtype)))?;
                if value.ty != Ty::Scalar(dtype) {
                    self.error(value.span, format!("`atomic` value must be {}, found {}", dtype.name(), value.ty));
                    return None;
                }
                Some(Expr { kind: ExprKind::Builtin { name: b, args: vec![v, value] }, ty: Ty::Void, sym: None, span })
            }
            Builtin::Reduce => {
                positional(self, args, 3, "reduce")?;
                let t = self.expr(&args[0].value, None)?;
                let Ty::Tile(ts) = &t.ty else {
                    self.error(t.span, format!("`reduce` needs a tile, found {}", t.ty));
                    return None;
                };
                let axis = self.expr(&args[1].value, Some(&Ty::Scalar(DType::I32)))?;
                let Some(axis_v) = axis.sym.as_ref().and_then(|s| s.as_constant()) else {
                    self.error(axis.span, "`reduce` axis must be a constant");
                    return None;
                };
                if axis_v < 0 || axis_v as usize >= ts.shape.len() {
                    self.error(axis.span, format!("axis {axis_v} is out of range for rank {}", ts.shape.len()));
                    return None;
                }
                let A::Name(opname) = &args[2].value.kind else {
                    self.error(args[2].value.span, "`reduce` needs an operation: sum, max, min or argmax");
                    return None;
                };
                let Some(op) = ReduceOp::from_name(&opname.name) else {
                    self.error(opname.span, format!("`{}` is not a reduction; use sum, max, min or argmax", opname.name));
                    return None;
                };
                let elem = match op {
                    ReduceOp::Argmax => DType::I32,
                    _ => ts.elem.read_dtype().unwrap_or(DType::F32),
                };
                let mut shape = ts.shape.clone();
                shape.remove(axis_v as usize);
                let ty = if shape.is_empty() { Ty::Scalar(elem) } else { Ty::Tile(Shaped::new(shape, Elem::Dtype(elem))) };
                let op_expr = Expr { kind: ExprKind::Int(op as i64), ty: Ty::Scalar(DType::I32), sym: None, span: opname.span };
                Some(Expr { kind: ExprKind::Builtin { name: b, args: vec![t, axis, op_expr] }, ty, sym: None, span })
            }
            Builtin::Extent => {
                positional(self, args, 2, "extent")?;
                let t = self.expr(&args[0].value, None)?;
                let Some(s) = t.ty.shaped().cloned() else {
                    self.error(t.span, format!("`extent` needs a tensor or tile, found {}", t.ty));
                    return None;
                };
                let axis = self.expr(&args[1].value, Some(&Ty::Scalar(DType::I32)))?;
                let Some(axis_v) = axis.sym.as_ref().and_then(|x| x.as_constant()) else {
                    self.error(axis.span, "`extent` axis must be a constant");
                    return None;
                };
                let Some(dim) = s.shape.get(axis_v.max(0) as usize) else {
                    self.error(axis.span, format!("axis {axis_v} is out of range for rank {}", s.shape.len()));
                    return None;
                };
                Some(Expr { kind: ExprKind::Builtin { name: b, args: vec![t, axis] }, ty: Ty::Scalar(DType::I32), sym: Some(dim.clone()), span })
            }
            Builtin::Fma => {
                positional(self, args, 3, "fma")?;
                let hint = match expected {
                    Some(Ty::Scalar(d)) if d.is_float() => Some(Ty::Scalar(*d)),
                    _ => None,
                };
                let a = self.expr(&args[0].value, hint.as_ref())?;
                let d = match a.ty {
                    Ty::Scalar(d) if d.is_float() => d,
                    _ => {
                        self.error(a.span, format!("`fma` needs float scalars, found {}", a.ty));
                        return None;
                    }
                };
                let b_ = self.expr(&args[1].value, Some(&Ty::Scalar(d)))?;
                let c = self.expr(&args[2].value, Some(&Ty::Scalar(d)))?;
                let mut dt = d;
                for x in [&b_, &c] {
                    match x.ty {
                        Ty::Scalar(e) if e.is_float() => dt = DType::promote(dt, e).unwrap(),
                        _ => {
                            self.error(x.span, format!("`fma` needs float scalars, found {}", x.ty));
                            return None;
                        }
                    }
                }
                Some(Expr { kind: ExprKind::Builtin { name: b, args: vec![a, b_, c] }, ty: Ty::Scalar(dt), sym: None, span })
            }
            Builtin::Exp | Builtin::ExpFast | Builtin::Rsqrt | Builtin::Sqrt | Builtin::Log | Builtin::Sin | Builtin::Cos | Builtin::Abs => {
                positional(self, args, 1, name_of(b))?;
                let a = self.expr(&args[0].value, expected)?;
                let ok = match a.ty {
                    Ty::Scalar(d) => d.is_float() || (b == Builtin::Abs && d.is_numeric()),
                    _ => false,
                };
                if !ok {
                    self.error(a.span, format!("`{}` needs a float scalar, found {}", name_of(b), a.ty));
                    return None;
                }
                let ty = a.ty.clone();
                Some(Expr { kind: ExprKind::Builtin { name: b, args: vec![a] }, ty, sym: None, span })
            }
            Builtin::Max | Builtin::Min => {
                positional(self, args, 2, name_of(b))?;
                let x = self.expr(&args[0].value, expected)?;
                let hint = match &x.ty {
                    Ty::Scalar(d) => Some(Ty::Scalar(*d)),
                    _ => None,
                };
                let y = self.expr(&args[1].value, hint.as_ref())?;
                let (Ty::Scalar(a), Ty::Scalar(c)) = (&x.ty, &y.ty) else {
                    self.error(span, format!("`{}` needs two numeric scalars", name_of(b)));
                    return None;
                };
                let Some(d) = DType::promote(*a, *c) else {
                    self.error(span, format!("`{}` between {} and {} needs an explicit cast", name_of(b), a.name(), c.name()));
                    return None;
                };
                Some(Expr { kind: ExprKind::Builtin { name: b, args: vec![x, y] }, ty: Ty::Scalar(d), sym: None, span })
            }
        }
    }

    fn intrinsic_call(&mut self, intr: &Intrinsic, args: &[ast::Arg], span: Span) -> Option<Expr> {
        if args.len() != intr.params.len() || args.iter().any(|a| a.name.is_some()) {
            self.error(span, format!("`{}` takes {} positional argument(s)", intr.name, intr.params.len()));
            return None;
        }
        let mut out = Vec::new();
        let mut float_dtype: Option<DType> = None;
        let mut named_dtype: Option<DType> = None;
        for (arg, p) in args.iter().zip(&intr.params) {
            match p {
                IntrinsicParam::DTypeName => {
                    let A::Name(n) = &arg.value.kind else {
                        self.error(arg.value.span, "expected a dtype name");
                        return None;
                    };
                    let Some(d) = DType::from_name(&n.name) else {
                        self.error(n.span, format!("`{}` is not a dtype", n.name));
                        return None;
                    };
                    named_dtype = Some(d);
                    out.push(Expr { kind: ExprKind::Int(0), ty: Ty::Scalar(d), sym: None, span: arg.value.span });
                }
                IntrinsicParam::FloatScalar => {
                    let e = self.expr(&arg.value, float_dtype.map(Ty::Scalar).as_ref())?;
                    let Ty::Scalar(d) = e.ty else {
                        self.error(e.span, format!("`{}` needs a float scalar, found {}", intr.name, e.ty));
                        return None;
                    };
                    if !d.is_float() {
                        self.error(e.span, format!("`{}` needs a float scalar, found {}", intr.name, d.name()));
                        return None;
                    }
                    float_dtype = Some(d);
                    out.push(e);
                }
                IntrinsicParam::Frag8x8 => {
                    let e = self.expr(&arg.value, None)?;
                    if !matches!(&e.ty, Ty::Frag(s) if s.shape == vec![Sym::constant(8), Sym::constant(8)]) {
                        self.error(e.span, format!("`{}` needs an 8x8 fragment, found {}", intr.name, e.ty));
                        return None;
                    }
                    out.push(e);
                }
                IntrinsicParam::Tile2 => {
                    let e = self.expr(&arg.value, None)?;
                    if !matches!(&e.ty, Ty::Tile(s) if s.shape.len() == 2) {
                        self.error(e.span, format!("`{}` needs a rank-2 tile, found {}", intr.name, e.ty));
                        return None;
                    }
                    out.push(e);
                }
                IntrinsicParam::Int => {
                    let e = self.expr(&arg.value, Some(&Ty::Scalar(DType::I32)))?;
                    if e.sym.is_none() {
                        self.error(e.span, format!("`{}` needs a static integer offset", intr.name));
                        return None;
                    }
                    out.push(e);
                }
            }
        }
        // Block loads and stores read an 8x8 window at (row, col): prove it fits.
        if matches!(intr.name, "simdgroup_load" | "simdgroup_load_t" | "simdgroup_store") {
            let Ty::Tile(s) = out[1].ty.clone() else { unreachable!() };
            for (k, dim) in s.shape.iter().enumerate() {
                let off = out[2 + k].sym.clone().unwrap();
                let iv = self.prover().interval(&off);
                self.require_nonneg(&iv.lo, out[2 + k].span, "block offset may be negative");
                self.require_nonneg(&dim.sub(&iv.hi).sub(&Sym::constant(8)), out[2 + k].span, &format!("8-wide block may exceed extent `{dim}`"));
            }
        }
        let ty = match intr.result {
            IntrinsicResult::Void => Ty::Void,
            IntrinsicResult::FloatScalar => Ty::Scalar(float_dtype.unwrap_or(DType::F32)),
            IntrinsicResult::Frag8x8OfNamedDtype => intrinsics::frag8x8(named_dtype.unwrap_or(DType::F32)),
        };
        Some(Expr { kind: ExprKind::Intrinsic { name: intr.name.to_string(), args: out }, ty, sym: None, span })
    }

    /// Parallel independence: every enclosing parallel index must select a distinct axis of the
    /// stored view, either as a point index `v` (or `v * c + e` without other parallel indices)
    /// or as a slice `v * c : (v + 1) * c`.
    /// How many work items a `parallel` block has, per index, derived from the regions its
    /// body stores. A point index on an axis of extent E gives E items; a slice
    /// `v * c : (v + 1) * c` on an axis of extent E gives E / c. Every store that an index
    /// selects must agree, so the kernel states what it computes and never how the work is cut.
    fn derive_extents(&mut self, vars: &[VarId], body: &[Stmt], span: Span) -> Vec<Sym> {
        let atoms: Vec<Atom> = vars
            .iter()
            .map(|v| match &self.vars[*v].kind {
                VarKind::Index(a) => a.clone(),
                _ => unreachable!(),
            })
            .collect();
        let mut found: Vec<Option<Sym>> = vec![None; vars.len()];
        let mut stores = Vec::new();
        collect_stored_views(body, &mut stores);
        for view in &stores {
            for (i, atom) in atoms.iter().enumerate() {
                let Some(extent) = self.extent_from_view(view, atom) else { continue };
                match &found[i] {
                    None => found[i] = Some(extent),
                    Some(prev) if self.prover().zero(&prev.clone().sub(&extent)) => {}
                    Some(prev) => {
                        let name = self.vars[vars[i]].name.clone();
                        self.error(view.span, format!("work-item count for `{name}` is `{prev}` from one store and `{extent}` from another"));
                    }
                }
            }
        }
        found
            .into_iter()
            .enumerate()
            .map(|(i, e)| match e {
                Some(e) => e,
                None => {
                    let name = self.vars[vars[i]].name.clone();
                    self.error(span, format!("`{name}` does not select a region of anything the body stores, so its work-item count cannot be derived"));
                    Sym::constant(1)
                }
            })
            .collect()
    }

    /// The extent `atom` ranges over, if this stored view selects one of its axes by it.
    fn extent_from_view(&mut self, view: &Expr, atom: &Atom) -> Option<Sym> {
        let atoms = [atom.clone()];
        let mut node = view;
        let mut per_axis: Vec<Option<(Option<Atom>, Option<Sym>)>> = Vec::new();
        loop {
            match &node.kind {
                ExprKind::Index { base, indices } => {
                    let mut here = Vec::new();
                    for idx in indices {
                        here.push(Some(match idx {
                            // A point index covers one element of its axis.
                            Index::Point(e) => (e.sym.as_ref().and_then(|s| selects_one(s, &atoms)), Some(Sym::constant(1))),
                            // A slice `v * c : (v + 1) * c` covers `c` elements of its axis.
                            Index::Slice { start: Some(st), end: Some(en) } => match (&st.sym, &en.sym) {
                                (Some(a), Some(b)) => (slice_selects(a, b, &atoms), Some(b.clone().sub(a))),
                                _ => (None, None),
                            },
                            _ => (None, None),
                        }));
                    }
                    here.extend(per_axis.drain(..));
                    per_axis = here;
                    node = base;
                }
                ExprKind::Transpose(inner) => node = inner,
                _ => break,
            }
        }
        let ExprKind::Var(v) = node.kind else { return None };
        let shape = self.vars[v].ty.shaped()?.shape.clone();
        for (axis, entry) in per_axis.iter().enumerate() {
            let Some((Some(a), Some(step))) = entry else { continue };
            if a != atom || axis >= shape.len() {
                continue;
            }
            // The axis is tiled by this index in steps of `step`, so it takes extent / step
            // values. A step that is itself a quotient of the extent, as `H / KV` is of `H`,
            // gives back the divisor: the tiling is what makes the division exact.
            let extent = shape[axis].clone();
            let count = match single_param_atom(step) {
                Some(Atom::Quot(n, d)) if *n == extent => (*d).clone(),
                _ => extent.quot(step),
            };
            return Some(count);
        }
        None
    }

    fn check_independent(&mut self, view: &Expr) {
        let Some(pvars) = self.parallel_vars.last().cloned() else { return };
        let atoms: Vec<Atom> = pvars
            .iter()
            .map(|v| match &self.vars[*v].kind {
                VarKind::Index(a) => a.clone(),
                _ => unreachable!(),
            })
            .collect();
        // Collect the index list of the outermost Index node on the view.
        let mut node = view;
        let mut per_axis: Vec<Option<Atom>> = Vec::new();
        loop {
            match &node.kind {
                ExprKind::Index { base, indices } => {
                    for idx in indices {
                        per_axis.push(match idx {
                            Index::Point(e) => e.sym.as_ref().and_then(|s| selects_one(s, &atoms)),
                            Index::Slice { start: Some(s), end: Some(e) } => match (&s.sym, &e.sym) {
                                (Some(a), Some(b)) => slice_selects(a, b, &atoms),
                                _ => None,
                            },
                            _ => None,
                        });
                    }
                    node = base;
                }
                ExprKind::Transpose(inner) => node = inner,
                _ => break,
            }
        }
        for (v, a) in pvars.iter().zip(&atoms) {
            let count = per_axis.iter().filter(|x| x.as_ref() == Some(a)).count();
            if count == 0 {
                let name = self.vars[*v].name.clone();
                self.error(view.span, format!("cannot prove work items are independent: parallel index `{name}` does not select a distinct region of the stored view; use a point index or a `{name} * c : ({name} + 1) * c` slice, or `atomic`"));
            }
        }
    }

    /// For lowerings: the region each written parameter covers must be the whole parameter,
    /// and the region each read parameter covers must be the whole parameter too (a body
    /// that skips part of an operand computes something else). Otherwise the block only
    /// applies where the uncovered part is empty.
    fn output_coverage(&mut self, stmts: &[Stmt]) {
        let mut writes: HashMap<VarId, Vec<Vec<(Sym, Sym)>>> = HashMap::new();
        collect_writes(stmts, &mut writes, &self.vars);
        let mut reads: HashMap<VarId, Vec<Vec<(Sym, Sym)>>> = HashMap::new();
        collect_reads(stmts, &mut reads, &self.vars);
        for (id, regions) in reads {
            writes.entry(id).or_default().extend(regions);
        }
        for (id, regions) in writes {
            let Ty::Tile(s) = self.vars[id].ty.clone() else { continue };
            for (axis, extent) in s.shape.iter().enumerate() {
                // hi of the union over regions along this axis
                let mut best_hi: Option<Sym> = None;
                let mut best_lo: Option<Sym> = None;
                for r in &regions {
                    let (lo, hi) = &r[axis];
                    let hi_iv = self.prover().interval(hi);
                    let lo_iv = self.prover().interval(lo);
                    best_hi = Some(match best_hi {
                        None => hi_iv.hi.clone(),
                        Some(b) if self.prover().le(&b, &hi_iv.hi) => hi_iv.hi.clone(),
                        Some(b) => b,
                    });
                    best_lo = Some(match best_lo {
                        None => lo_iv.lo.clone(),
                        Some(b) if self.prover().le(&lo_iv.lo, &b) => lo_iv.lo.clone(),
                        Some(b) => b,
                    });
                }
                let (Some(hi), Some(lo)) = (best_hi, best_lo) else { continue };
                // Require lo == 0 and hi == extent - 1; record the shortfall as a residual.
                let top = extent.sub(&Sym::constant(1)).sub(&hi);
                if !self.prover().zero(&top) {
                    // top >= 0 always holds (bounds were checked); the block needs top == 0, i.e. -top >= 0.
                    let need = top.neg();
                    if !self.residual.contains(&need) {
                        self.residual.push(need);
                    }
                }
                if !self.prover().zero(&lo) {
                    let need = lo.neg();
                    if !self.residual.contains(&need) {
                        self.residual.push(need);
                    }
                }
            }
        }
    }
}

fn name_of(b: Builtin) -> &'static str {
    match b {
        Builtin::Load => "load",
        Builtin::Store => "store",
        Builtin::Atomic => "atomic",
        Builtin::Reduce => "reduce",
        Builtin::Extent => "extent",
        Builtin::Fma => "fma",
        Builtin::Exp => "exp",
        Builtin::ExpFast => "exp_fast",
        Builtin::Rsqrt => "rsqrt",
        Builtin::Sqrt => "sqrt",
        Builtin::Log => "log",
        Builtin::Sin => "sin",
        Builtin::Cos => "cos",
        Builtin::Abs => "abs",
        Builtin::Max => "max",
        Builtin::Min => "min",
    }
}

fn promote_elem(a: &Elem, b: &Elem) -> DType {
    match (a, b) {
        (Elem::Dtype(x), Elem::Dtype(y)) => DType::promote(*x, *y).unwrap_or(DType::F32),
        _ => DType::F32,
    }
}

fn single_param(s: &Sym) -> Option<String> {
    let params = s.params();
    if params.len() == 1 && *s == Sym::param(&params[0]) {
        Some(params[0].clone())
    } else {
        None
    }
}

/// `v` or `v * c + e` where `e` contains no parallel atom: which parallel atom it selects.
fn selects_one(s: &Sym, atoms: &[Atom]) -> Option<Atom> {
    let present: Vec<&Atom> = atoms.iter().filter(|a| s.atoms().contains(a)).collect();
    if present.len() != 1 {
        return None;
    }
    Some(present[0].clone())
}

/// The single atom of a symbol that is exactly one atom with coefficient 1.
fn single_param_atom(s: &Sym) -> Option<Atom> {
    let atoms = s.atoms();
    if let [a] = atoms.as_slice() {
        if *s == Sym::atom(a.clone()) {
            return Some(a.clone());
        }
    }
    None
}

/// Check a lowering's restated signature against its construct, returning the specialized
/// signature and the element substitutions it makes. `None` on any mismatch.
fn specialize(sig: &Signature, l: &ast::LowerDecl, diagnostics: &mut Vec<Diagnostic>) -> Option<(Signature, Vec<(String, Elem)>)> {
    let mut bad = false;
    if l.shape.len() != sig.shape_params.len() || l.shape.iter().zip(&sig.shape_params).any(|(a, b)| &a.name != b) {
        let want = sig.shape_params.join(", ");
        let span = l.shape.first().map(|i| i.span).unwrap_or(l.name.span);
        diagnostics.push(Diagnostic::new(span, format!("`{}` declares shape parameters [{want}]; a lowering restates them unchanged", sig.name)));
        return None;
    }
    if l.params.len() != sig.params.len() {
        diagnostics.push(Diagnostic::new(l.name.span, format!("`{}` takes {} parameters, this lowering restates {}", sig.name, sig.params.len(), l.params.len())));
        return None;
    }
    let shape_params: Vec<String> = sig.shape_params.clone();
    let mut specialized = sig.clone();
    let mut bindings: Vec<(String, Elem)> = Vec::new();
    for (i, (want_name, want_ty)) in sig.params.iter().enumerate() {
        let got = &l.params[i];
        if &got.name.name != want_name {
            diagnostics.push(Diagnostic::new(got.name.span, format!("parameter {} of `{}` is `{want_name}`", i + 1, sig.name)));
            bad = true;
            continue;
        }
        // Index parameters are restated as `index[E]` and carry no element type.
        if got.ty.head.name == "index" {
            if !sig.index_params.iter().any(|(n, _)| n == want_name) {
                diagnostics.push(Diagnostic::new(got.ty.span, format!("`{want_name}` is not an index parameter of `{}`", sig.name)));
                bad = true;
            }
            continue;
        }
        let mut elems = Vec::new();
        let got_ty = match type_from_ast(&got.ty, &shape_params, &mut elems) {
            Ok(t) => t,
            Err(d) => {
                diagnostics.push(d);
                bad = true;
                continue;
            }
        };
        match (want_ty, &got_ty) {
            (Ty::Scalar(a), Ty::Scalar(b)) if a == b => {}
            (Ty::Tensor(w), Ty::Tensor(g)) | (Ty::Tile(w), Ty::Tile(g)) => {
                if w.shape != g.shape {
                    diagnostics.push(Diagnostic::new(got.ty.span, format!("`{want_name}` has shape {} in `{}`", shape_text(&w.shape), sig.name)));
                    bad = true;
                    continue;
                }
                match (&w.elem, &g.elem) {
                    // Unchanged.
                    (a, b) if a == b => {}
                    // A concrete element where the construct has a parameter: a specialization.
                    (Elem::Param(p), concrete @ (Elem::Dtype(_) | Elem::Repr(_))) => {
                        match bindings.iter().find(|(n, _)| n == p) {
                            Some((_, prev)) if prev != concrete => {
                                diagnostics.push(Diagnostic::new(got.ty.span, format!("`{p}` is specialized to two different types in one lowering")));
                                bad = true;
                                continue;
                            }
                            Some(_) => {}
                            None => bindings.push((p.clone(), concrete.clone())),
                        }
                        for (_, ty) in specialized.params.iter_mut() {
                            if let Ty::Tensor(s) | Ty::Tile(s) = ty {
                                if s.elem == Elem::Param(p.clone()) {
                                    s.elem = concrete.clone();
                                    if matches!(concrete, Elem::Repr(_)) {
                                        s.packed_axis = Some(s.shape.len() - 1);
                                    }
                                }
                            }
                        }
                    }
                    (a, b) => {
                        diagnostics.push(Diagnostic::new(got.ty.span, format!("`{want_name}` has element type {a} in `{}`; a lowering may only replace an element parameter with a concrete type, not {b}", sig.name)));
                        bad = true;
                    }
                }
            }
            (a, b) => {
                diagnostics.push(Diagnostic::new(got.ty.span, format!("`{want_name}` is {a} in `{}`, restated as {b}", sig.name)));
                bad = true;
            }
        }
    }
    if bad {
        return None;
    }
    Some((specialized, bindings))
}

fn shape_text(shape: &[Sym]) -> String {
    format!("[{}]", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
}

/// The views a body stores into, in order.
fn collect_stored_views(stmts: &[Stmt], out: &mut Vec<Expr>) {
    for s in stmts {
        match &s.kind {
            StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => collect_stored_views(body, out),
            StmtKind::If { then, els, .. } => {
                collect_stored_views(then, out);
                collect_stored_views(els, out);
            }
            StmtKind::Expr(e) => {
                if let ExprKind::Builtin { name: Builtin::Store | Builtin::Atomic, args } = &e.kind {
                    if let Some(v) = args.get(1) {
                        out.push(v.clone());
                    }
                }
            }
            StmtKind::Assign { .. } => {}
        }
    }
}

/// A slice `v * c : (v + 1) * c` selects `v`.
fn slice_selects(start: &Sym, end: &Sym, atoms: &[Atom]) -> Option<Atom> {
    let a = selects_one(start, atoms)?;
    if selects_one(end, atoms)? != a {
        return None;
    }
    // end - start must not depend on v: substituting v := v + 1 into start must give end.
    let shifted = start.subst(&a, &Sym::atom(a.clone()).add(&Sym::constant(1)));
    if shifted == *end {
        Some(a)
    } else {
        None
    }
}

/// Does `body` contain `name[targets...] = ...` at its top level?
/// Bounds on single atoms implied by a condition (or by its negation): `x < e` gives
/// `x <= e - 1`, `e < x` gives `x >= e + 1`, and so on. `and` contributes both sides when
/// not negated; `or` contributes both sides only when negated.
fn path_bounds(cond: &Expr, negate: bool) -> Vec<(Atom, Option<Sym>, Option<Sym>)> {
    let ExprKind::Binary { op, lhs, rhs } = &cond.kind else { return Vec::new() };
    match (op, negate) {
        (BinaryOp::And, false) | (BinaryOp::Or, true) => {
            let mut out = path_bounds(lhs, negate);
            out.extend(path_bounds(rhs, negate));
            return out;
        }
        (BinaryOp::And, true) | (BinaryOp::Or, false) => return Vec::new(),
        _ => {}
    }
    let (Some(l), Some(r)) = (&lhs.sym, &rhs.sym) else { return Vec::new() };
    // Normalize to `l <= r` / `l >= r` forms as (lo_side, hi_side) meaning lo_side <= hi_side.
    let one = Sym::constant(1);
    let le: Option<(Sym, Sym)> = match (op, negate) {
        (BinaryOp::Lt, false) | (BinaryOp::Ge, true) => Some((l.clone(), r.sub(&one))),
        (BinaryOp::Le, false) | (BinaryOp::Gt, true) => Some((l.clone(), r.clone())),
        (BinaryOp::Gt, false) | (BinaryOp::Le, true) => Some((r.clone(), l.sub(&one))),
        (BinaryOp::Ge, false) | (BinaryOp::Lt, true) => Some((r.clone(), l.clone())),
        _ => None,
    };
    let Some((small, big)) = le else { return Vec::new() };
    let mut out = Vec::new();
    if let Some(a) = single_atom(&small) {
        if !big.atoms().contains(&a) {
            out.push((a, None, Some(big.clone())));
        }
    }
    if let Some(a) = single_atom(&big) {
        if !small.atoms().contains(&a) {
            out.push((a, Some(small.clone()), None));
        }
    }
    out
}

fn single_atom(s: &Sym) -> Option<Atom> {
    let atoms = s.atoms();
    if let [a] = atoms.as_slice() {
        if *s == Sym::atom(a.clone()) {
            return Some(a.clone());
        }
    }
    None
}

/// Whether every path through `body` assigns `name[targets...]`: a direct element assignment
/// with the loop's own indices, or an `if` whose branches both do.
fn body_assigns_all(body: &ast::Block, name: &str, targets: &[Ident]) -> bool {
    body.stmts.iter().any(|s| match &s.kind {
        ast::StmtKind::Assign { target, op: AssignOp::Assign, .. } => match &target.kind {
            A::Index { base, indices } => {
                matches!(&base.kind, A::Name(n) if n.name == name)
                    && indices.len() == targets.len()
                    && indices.iter().zip(targets).all(|(i, t)| matches!(i, ast::Index::Expr(e) if matches!(&e.kind, A::Name(n) if n.name == t.name)))
            }
            _ => false,
        },
        ast::StmtKind::If { then, els: Some(els), .. } => body_assigns_all(then, name, targets) && body_assigns_all(els, name, targets),
        _ => false,
    })
}

/// Collect the written regions of parameter tiles: per write, per axis an inclusive (lo, hi).
fn collect_writes(stmts: &[Stmt], out: &mut HashMap<VarId, Vec<Vec<(Sym, Sym)>>>, vars: &[Var]) {
    for s in stmts {
        match &s.kind {
            StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => collect_writes(body, out, vars),
            StmtKind::If { then, els, .. } => {
                collect_writes(then, out, vars);
                collect_writes(els, out, vars);
            }
            StmtKind::Assign { target, .. } => {
                if let ExprKind::Index { base, indices } = &target.kind {
                    if let ExprKind::Var(v) = base.kind {
                        if matches!(vars[v].kind, VarKind::Param(_)) {
                            let mut region = Vec::new();
                            for i in indices {
                                if let Index::Point(e) = i {
                                    if let Some(sym) = &e.sym {
                                        region.push((sym.clone(), sym.clone()));
                                    }
                                }
                            }
                            if region.len() == indices.len() {
                                out.entry(v).or_default().push(region);
                            }
                        }
                    }
                }
            }
            StmtKind::Expr(e) => {
                if let ExprKind::Intrinsic { name, args } = &e.kind {
                    if name == "simdgroup_store" {
                        if let ExprKind::Var(v) = args[1].kind {
                            if matches!(vars[v].kind, VarKind::Param(_)) {
                                let r = args[2].sym.clone().unwrap();
                                let c = args[3].sym.clone().unwrap();
                                out.entry(v).or_default().push(vec![(r.clone(), r.add(&Sym::constant(7))), (c.clone(), c.add(&Sym::constant(7)))]);
                            }
                        }
                    }
                }
            }
        }
    }
}


/// Collect the read regions of parameter tiles: element reads and intrinsic block loads.
fn collect_reads(stmts: &[Stmt], out: &mut HashMap<VarId, Vec<Vec<(Sym, Sym)>>>, vars: &[Var]) {
    fn expr(e: &Expr, out: &mut HashMap<VarId, Vec<Vec<(Sym, Sym)>>>, vars: &[Var]) {
        match &e.kind {
            ExprKind::Index { base, indices } => {
                if let ExprKind::Var(v) = base.kind {
                    if matches!(vars[v].kind, VarKind::Param(_)) && matches!(vars[v].ty, Ty::Tile(_)) {
                        let mut region = Vec::new();
                        for i in indices {
                            if let Index::Point(p) = i {
                                if let Some(sym) = &p.sym {
                                    region.push((sym.clone(), sym.clone()));
                                }
                            }
                        }
                        if region.len() == indices.len() {
                            out.entry(v).or_default().push(region);
                        }
                    }
                }
                for i in indices {
                    if let Index::Point(p) = i {
                        expr(p, out, vars);
                    }
                }
            }
            ExprKind::Intrinsic { name, args } if name == "simdgroup_load" || name == "simdgroup_load_t" => {
                if let ExprKind::Var(v) = args[1].kind {
                    if matches!(vars[v].kind, VarKind::Param(_)) {
                        let r = args[2].sym.clone().unwrap();
                        let c = args[3].sym.clone().unwrap();
                        out.entry(v).or_default().push(vec![(r.clone(), r.add(&Sym::constant(7))), (c.clone(), c.add(&Sym::constant(7)))]);
                    }
                }
            }
            ExprKind::Builtin { args, .. } | ExprKind::Intrinsic { args, .. } | ExprKind::Call { args, .. } | ExprKind::Tuple(args) => {
                for a in args {
                    expr(a, out, vars);
                }
            }
            ExprKind::Unary { expr: x, .. } | ExprKind::Cast { expr: x, .. } | ExprKind::Transpose(x) | ExprKind::Accessor { base: x, .. } | ExprKind::Lanes { base: x, .. } => expr(x, out, vars),
            ExprKind::Binary { lhs, rhs, .. } => {
                expr(lhs, out, vars);
                expr(rhs, out, vars);
            }
            _ => {}
        }
    }
    for s in stmts {
        match &s.kind {
            StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => collect_reads(body, out, vars),
            StmtKind::If { cond, then, els } => {
                expr(cond, out, vars);
                collect_reads(then, out, vars);
                collect_reads(els, out, vars);
            }
            StmtKind::Assign { target, value, .. } => {
                if let ExprKind::Index { indices, .. } = &target.kind {
                    for i in indices {
                        if let Index::Point(p) = i {
                            expr(p, out, vars);
                        }
                    }
                }
                expr(value, out, vars);
            }
            StmtKind::Expr(e) => expr(e, out, vars),
        }
    }
}
