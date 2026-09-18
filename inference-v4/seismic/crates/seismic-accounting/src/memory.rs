//! Checked-IR access-region derivation. These are touched backing bytes, not traffic
//! on a selected physical path. Residency and reaching definitions are separate.
//!
//! This initial exact evaluator visits only tensor-accessing loop bodies, under an
//! explicit analysis budget. Unsupported/dynamic accesses retain missing coverage;
//! neither allocation size nor an iteration sample is substituted for their union.

mod affine;
use crate::region::Accesses;
use seismic_lang::{
    ast::{BinaryOp, UnaryOp},
    ir::*,
    program::Program,
    repr,
    sym::{Atom, Sym},
    types::{Elem, Ty},
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Backing {
    pub identity: String,
    pub byte_offset: u64,
}

/// Parameter/plane -> backing. Dense tensors use the empty plane name. Packed
/// tensors use `words`, `scale`, `bias`. Omitted bindings get distinct identities.
pub type Bindings = HashMap<(String, String), Backing>;

#[derive(Clone, Debug, Default)]
pub struct MemoryAccount {
    pub accesses: Accesses,
    pub unavailable: Vec<String>,
    pub analysis_steps: usize,
}

impl MemoryAccount {
    pub fn is_exact(&self) -> bool {
        self.unavailable.is_empty()
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone)]
struct View {
    parameter: String,
    elem: Elem,
    shape: Vec<u64>,
    strides: Vec<u64>,
    offset: u64,
}

#[derive(Clone, Default)]
struct Frame {
    env: HashMap<String, i64>,
    views: HashMap<VarId, View>,
}

struct Walker<'a> {
    program: &'a Program,
    bindings: &'a Bindings,
    account: MemoryAccount,
    remaining: usize,
    calls: Vec<String>,
}

pub fn derive(
    program: &Program,
    name: &str,
    shapes: &HashMap<String, i64>,
    bindings: &Bindings,
    budget: usize,
) -> Result<MemoryAccount, String> {
    derive_specialized(program, name, shapes, &HashMap::new(), bindings, budget)
}

pub fn derive_specialized(
    program: &Program,
    name: &str,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
    bindings: &Bindings,
    budget: usize,
) -> Result<MemoryAccount, String> {
    let f = program
        .functions
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| format!("no function `{name}`"))?;
    seismic_lang::program::validate_element_bindings(f, elements)?;
    let mut frame = Frame {
        env: shapes.clone(),
        views: HashMap::new(),
    };
    for p in &f.shape_params {
        if shapes.get(p).is_none_or(|n| *n < 0) {
            return Err(format!("unbound or negative shape `{p}`"));
        }
    }
    for (id, (parameter, ty)) in f.params.iter().enumerate() {
        if let Ty::Tensor(sh) = ty {
            let shape = sh
                .shape
                .iter()
                .map(|s| extent(s, &frame))
                .collect::<Result<Vec<_>, _>>()?;
            let elem = match &sh.elem {
                Elem::Param(p) => elements[p].clone(),
                other => other.clone(),
            };
            if let Elem::Repr(name) = &elem {
                let r =
                    repr::lookup(name).ok_or_else(|| format!("unknown representation `{name}`"))?;
                if shape.last().is_none_or(|n| n % u64::from(r.group) != 0) {
                    return Err(format!("packed tensor `{parameter}` requires complete groups of {} on its last axis", r.group));
                }
            }
            let mut strides = vec![1; shape.len()];
            let mut stride = 1u64;
            for i in (0..shape.len()).rev() {
                strides[i] = stride;
                stride = stride.checked_mul(shape[i]).ok_or("tensor size overflow")?;
            }
            frame.views.insert(
                id,
                View {
                    parameter: parameter.clone(),
                    elem,
                    shape,
                    strides,
                    offset: 0,
                },
            );
        }
    }
    let mut w = Walker {
        program,
        bindings,
        account: MemoryAccount::default(),
        remaining: budget,
        calls: vec![name.into()],
    };
    if let Err(error) = w.block(&f.body, f, &mut frame) {
        w.account.unavailable.push(error);
    }
    Ok(w.account)
}

fn extent(s: &Sym, f: &Frame) -> Result<u64, String> {
    s.eval(&|p| f.env.get(p).copied())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| format!("unresolved, negative or overflowing extent `{s}`"))
}

fn integer(e: &Expr, f: &Frame) -> Result<i64, String> {
    e.sym
        .as_ref()
        .and_then(|s| s.eval(&|p| f.env.get(p).copied()))
        .or({
            if let ExprKind::Int(n) = e.kind {
                Some(n)
            } else {
                None
            }
        })
        .ok_or_else(|| format!("runtime index at byte {}", e.span.start))
}

fn condition(e: &Expr, f: &Frame) -> Option<bool> {
    match &e.kind {
        ExprKind::Bool(b) => Some(*b),
        ExprKind::Unary {
            op: UnaryOp::Not,
            expr,
        } => Some(!condition(expr, f)?),
        ExprKind::Binary { op, lhs, rhs } => match op {
            BinaryOp::And => Some(condition(lhs, f)? && condition(rhs, f)?),
            BinaryOp::Or => Some(condition(lhs, f)? || condition(rhs, f)?),
            op => {
                let a = integer(lhs, f).ok()?;
                let b = integer(rhs, f).ok()?;
                Some(match op {
                    BinaryOp::Eq => a == b,
                    BinaryOp::Ne => a != b,
                    BinaryOp::Lt => a < b,
                    BinaryOp::Le => a <= b,
                    BinaryOp::Gt => a > b,
                    BinaryOp::Ge => a >= b,
                    _ => return None,
                })
            }
        },
        _ => None,
    }
}

fn view(e: &Expr, f: &Frame) -> Result<View, String> {
    match &e.kind {
        ExprKind::Builtin {
            name: Builtin::Reshape,
            args,
        } => {
            let mut source = view(&args[0], f)?;
            if matches!(source.elem, Elem::Repr(_)) {
                return Err("reshape currently requires dense storage".into());
            }
            let convert = |dims: &[u64]| {
                dims.iter()
                    .map(|n| i64::try_from(*n).map_err(|_| "reshape extent overflow".to_string()))
                    .collect::<Result<Vec<_>, _>>()
            };
            let target =
                e.ty.shaped()
                    .ok_or("reshape result is not shaped")?
                    .shape
                    .iter()
                    .map(|s| extent(s, f))
                    .collect::<Result<Vec<_>, _>>()?;
            let strides = seismic_lang::layout::reshape_strides(
                &convert(&source.shape)?,
                &convert(&source.strides)?,
                &convert(&target)?,
            )?;
            source.shape = target;
            source.strides = strides.into_iter().map(|n| n as u64).collect();
            Ok(source)
        }

        ExprKind::Var(v) => f
            .views
            .get(v)
            .cloned()
            .ok_or_else(|| format!("unresolved tensor view at byte {}", e.span.start)),
        ExprKind::Transpose(base) => {
            let mut v = view(base, f)?;
            let n = v.shape.len();
            if n < 2 {
                return Err("transpose rank is less than two".into());
            }
            v.shape.swap(n - 1, n - 2);
            v.strides.swap(n - 1, n - 2);
            Ok(v)
        }
        ExprKind::Index { base, indices } => {
            let source = view(base, f)?;
            if indices.len() > source.shape.len() {
                return Err("view index exceeds rank".into());
            }
            let mut result = View {
                shape: Vec::new(),
                strides: Vec::new(),
                ..source.clone()
            };
            for axis in 0..source.shape.len() {
                let size = source.shape[axis];
                let (start, end, keep) = match indices.get(axis) {
                    Some(Index::Point(e)) => {
                        let n = u64::try_from(integer(e, f)?).map_err(|_| "negative index")?;
                        (n, n.checked_add(1).ok_or("index overflow")?, false)
                    }
                    Some(Index::Slice { start, end }) => {
                        let start = start
                            .as_ref()
                            .map(|e| integer(e, f))
                            .transpose()?
                            .map(u64::try_from)
                            .transpose()
                            .map_err(|_| "negative slice start")?
                            .unwrap_or(0);
                        let end = end
                            .as_ref()
                            .map(|e| integer(e, f))
                            .transpose()?
                            .map(u64::try_from)
                            .transpose()
                            .map_err(|_| "negative slice end")?
                            .unwrap_or(size);
                        (start, end, true)
                    }
                    None => (0, size, true),
                };
                if start > end || end > size {
                    return Err("view lies outside backing tensor shape".into());
                }
                result.offset = result
                    .offset
                    .checked_add(
                        start
                            .checked_mul(source.strides[axis])
                            .ok_or("view offset overflow")?,
                    )
                    .ok_or("view offset overflow")?;
                if keep {
                    result.shape.push(end - start);
                    result.strides.push(source.strides[axis]);
                }
            }
            Ok(result)
        }
        _ => Err(format!("unsupported tensor view at byte {}", e.span.start)),
    }
}

// Determine whether a loop contains external tensor accesses. Pure tile arithmetic
// cannot change backing regions and must not cost an iteration per scalar multiply.
fn tensor_expr(e: &Expr) -> bool {
    if matches!(e.ty, Ty::Tensor(_)) {
        return true;
    }
    match &e.kind {
        ExprKind::Call { args, .. }
        | ExprKind::Builtin { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Tuple(args) => args.iter().any(tensor_expr),
        ExprKind::Index { base, indices } => {
            tensor_expr(base)
                || indices.iter().any(|i| match i {
                    Index::Point(e) => tensor_expr(e),
                    Index::Slice { start, end } => start.iter().chain(end).any(tensor_expr),
                })
        }
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } | ExprKind::Transpose(expr) => {
            tensor_expr(expr)
        }
        ExprKind::Binary { lhs, rhs, .. } => tensor_expr(lhs) || tensor_expr(rhs),
        ExprKind::Accessor { base, .. } | ExprKind::Lanes { base, .. } => tensor_expr(base),
        _ => false,
    }
}

fn tensor_body(body: &[Stmt]) -> bool {
    body.iter().any(|s| match &s.kind {
        StmtKind::Expr(e) => tensor_expr(e),
        StmtKind::Assign { target, value, .. } => tensor_expr(target) || tensor_expr(value),
        StmtKind::If { cond, then, els } => {
            tensor_expr(cond) || tensor_body(then) || tensor_body(els)
        }
        StmtKind::LoadLoop { .. } => true,
        StmtKind::Parallel { body, .. }
        | StmtKind::Owned { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. } => tensor_body(body),
    })
}

impl Walker<'_> {
    fn tick(&mut self) -> Result<(), String> {
        if self.remaining == 0 {
            return Err(
                "access-region analysis budget exhausted; remaining coverage unavailable".into(),
            );
        }
        self.remaining -= 1;
        self.account.analysis_steps += 1;
        Ok(())
    }
    fn block(&mut self, body: &[Stmt], f: &Function, frame: &mut Frame) -> Result<(), String> {
        for s in body {
            self.stmt(s, f, frame)
                .map_err(|error| format!("{}:{}: {error}", f.name, s.span.start))?;
        }
        Ok(())
    }
    fn loops(
        &mut self,
        vars: &[VarId],
        extents: &[(i64, i64)],
        body: &[Stmt],
        f: &Function,
        frame: &mut Frame,
    ) -> Result<(), String> {
        if vars.is_empty() {
            self.block(body, f, frame)?;
            return Ok(());
        }
        let VarKind::Index(Atom::Param(name)) = &f.vars[vars[0]].kind else {
            return Err("unresolved loop index".into());
        };
        if vars.len() == 1 {
            let before = self.remaining;
            let accesses = affine::sweep(
                body,
                frame,
                name,
                extents[0].0,
                extents[0].1,
                &mut self.remaining,
            );
            self.account.analysis_steps += before - self.remaining;
            if let Some(accesses) = accesses {
                for (view, mode) in accesses {
                    self.touch(&view, mode)?;
                }
                return Ok(());
            }
        }
        let previous = frame.env.get(name).copied();
        for i in extents[0].0..extents[0].1 {
            self.tick()?;
            frame.env.insert(name.clone(), i);
            self.loops(&vars[1..], &extents[1..], body, f, frame)?;
        }
        if let Some(value) = previous {
            frame.env.insert(name.clone(), value);
        } else {
            frame.env.remove(name);
        }
        Ok(())
    }
    fn stmt(&mut self, s: &Stmt, f: &Function, frame: &mut Frame) -> Result<(), String> {
        self.tick()?;
        match &s.kind {
            StmtKind::Parallel {
                vars,
                extents,
                body,
            } => {
                if tensor_body(body) {
                    let ext = extents
                        .iter()
                        .map(|e| {
                            extent(e, frame).and_then(|n| {
                                i64::try_from(n)
                                    .map(|n| (0, n))
                                    .map_err(|_| "loop extent overflow".into())
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.loops(vars, &ext, body, f, frame)?;
                }
            }
            StmtKind::Owned { vars, tile, body } => {
                if tensor_body(body) {
                    let sh = tile.ty.shaped().ok_or("owned tile has no shape")?;
                    let ext = sh
                        .shape
                        .iter()
                        .map(|e| {
                            extent(e, frame).and_then(|n| {
                                i64::try_from(n)
                                    .map(|n| (0, n))
                                    .map_err(|_| "loop extent overflow".into())
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.loops(vars, &ext, body, f, frame)?;
                }
            }
            StmtKind::Range { var, lo, hi, body } => {
                if tensor_body(body) {
                    let lo = lo
                        .eval(&|p| frame.env.get(p).copied())
                        .ok_or("runtime loop start")?;
                    let hi = hi
                        .eval(&|p| frame.env.get(p).copied())
                        .ok_or("runtime loop end")?;
                    self.loops(&[*var], &[(lo, hi)], body, f, frame)?;
                }
            }
            StmtKind::LoadLoop {
                views,
                axis,
                piece,
                body,
                ..
            } => {
                let mut n = None;
                for e in views {
                    self.expr(e, f, frame)?;
                    let v = view(e, frame)?;
                    self.touch(&v, Mode::Read)?;
                    let current = *v.shape.get(*axis).ok_or("stream axis out of range")?;
                    if n.is_some_and(|n| n != current) {
                        return Err("stream extents disagree".into());
                    }
                    n = Some(current);
                }
                if n == Some(0) {
                    return Ok(());
                }
                let Atom::Param(name) = piece else {
                    return Err("unresolved stream piece".into());
                };
                let mut inner = frame.clone();
                inner.env.insert(
                    name.clone(),
                    i64::try_from(n.ok_or("stream has no views")?)
                        .map_err(|_| "stream extent overflow")?,
                );
                self.block(body, f, &mut inner)?;
            }
            StmtKind::If { cond, then, els } => {
                self.expr(cond, f, frame)?;
                match condition(cond, frame) {
                    Some(true) => self.block(then, f, frame)?,
                    Some(false) => self.block(els, f, frame)?,
                    None => {
                        if tensor_body(then) || tensor_body(els) {
                            return Err(
                                "runtime branch: accessed regions remain conditional".into()
                            );
                        }
                    }
                }
            }
            StmtKind::Assign { target, value, op } => {
                self.expr(value, f, frame)?;
                if let ExprKind::Var(id) = target.kind {
                    if matches!(target.ty, Ty::Tensor(_)) {
                        frame.views.insert(id, view(value, frame)?);
                    }
                }
                if let ExprKind::Index { base, indices } = &target.kind {
                    for i in indices {
                        self.index(i, f, frame)?;
                    }
                    if matches!(base.ty, Ty::Tensor(_)) {
                        self.touch(
                            &view(target, frame)?,
                            if *op == seismic_lang::ast::AssignOp::Assign {
                                Mode::Write
                            } else {
                                Mode::ReadWrite
                            },
                        )?;
                    }
                }
            }
            StmtKind::Expr(e) => self.expr(e, f, frame)?,
            StmtKind::Lanes { .. } => {
                return Err("lane accesses require realization accounting".into())
            }
        }
        Ok(())
    }
    fn index(&mut self, i: &Index, f: &Function, frame: &mut Frame) -> Result<(), String> {
        match i {
            Index::Point(e) => self.expr(e, f, frame)?,
            Index::Slice { start, end } => {
                for e in start.iter().chain(end) {
                    self.expr(e, f, frame)?;
                }
            }
        }
        Ok(())
    }
    fn expr(&mut self, e: &Expr, f: &Function, frame: &mut Frame) -> Result<(), String> {
        match &e.kind {
            ExprKind::Call {
                callee,
                shape_args,
                args,
                ..
            } => {
                for arg in args {
                    self.expr(arg, f, frame)?;
                }
                if !args.iter().any(|a| matches!(a.ty, Ty::Tensor(_))) {
                    return Ok(());
                }
                let g = self
                    .program
                    .functions
                    .iter()
                    .find(|g| g.name == *callee)
                    .ok_or("unresolved call")?;
                if self.calls.contains(callee) {
                    return Err("recursive access derivation".into());
                }
                let mut inner = Frame::default();
                for (name, s) in g.shape_params.iter().zip(shape_args) {
                    inner.env.insert(
                        name.clone(),
                        s.eval(&|p| frame.env.get(p).copied())
                            .ok_or("runtime call shape")?,
                    );
                }
                for (id, arg) in args.iter().enumerate() {
                    if matches!(arg.ty, Ty::Tensor(_)) {
                        inner.views.insert(id, view(arg, frame)?);
                    }
                }
                self.calls.push(callee.clone());
                self.block(&g.body, g, &mut inner)?;
                self.calls.pop();
            }
            ExprKind::Load { view, .. } => {
                self.expr(view, f, frame)?;
                self.touch_expr(view, frame, Mode::Read)?;
            }
            ExprKind::Builtin { name, args } => {
                for arg in args {
                    self.expr(arg, f, frame)?;
                }
                match name {
                    Builtin::Load => self.touch_expr(&args[0], frame, Mode::Read)?,
                    Builtin::Store => self.touch_expr(&args[1], frame, Mode::Write)?,
                    Builtin::Atomic => self.touch_expr(&args[0], frame, Mode::ReadWrite)?,
                    _ => {}
                }
            }
            ExprKind::Index { base, indices } => {
                self.expr(base, f, frame)?;
                for i in indices {
                    self.index(i, f, frame)?;
                }
                if matches!(base.ty, Ty::Tensor(_)) && matches!(e.ty, Ty::Scalar(_)) {
                    self.touch(&view(e, frame)?, Mode::Read)?;
                }
            }
            ExprKind::Unary { expr, .. }
            | ExprKind::Cast { expr, .. }
            | ExprKind::Transpose(expr) => self.expr(expr, f, frame)?,
            ExprKind::Binary { op, lhs, rhs } => {
                self.expr(lhs, f, frame)?;
                if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    match condition(lhs, frame) {
                        Some(v) if (v && *op == BinaryOp::Or) || (!v && *op == BinaryOp::And) => {
                            return Ok(())
                        }
                        None if tensor_expr(rhs) => {
                            return Err("runtime short-circuit tensor access".into())
                        }
                        _ => {}
                    }
                }
                self.expr(rhs, f, frame)?;
            }
            ExprKind::Tuple(items) => {
                for i in items {
                    self.expr(i, f, frame)?;
                }
            }
            ExprKind::Intrinsic { .. } | ExprKind::Accessor { .. } | ExprKind::Lanes { .. } => {
                return Err("backend accesses require realization accounting".into())
            }
            _ => {}
        }
        Ok(())
    }
    fn touch_expr(&mut self, e: &Expr, frame: &Frame, mode: Mode) -> Result<(), String> {
        if let ExprKind::Tuple(items) = &e.kind {
            for i in items {
                self.touch_expr(i, frame, mode)?;
            }
        } else {
            self.touch(&view(e, frame)?, mode)?;
        }
        Ok(())
    }
    fn touch(&mut self, v: &View, mode: Mode) -> Result<(), String> {
        if v.shape.contains(&0) {
            return Ok(());
        }
        let mut contiguous = 1u64;
        let mut prefix = v.shape.len();
        while prefix > 0 && v.strides[prefix - 1] == contiguous {
            prefix -= 1;
            contiguous = contiguous
                .checked_mul(v.shape[prefix])
                .ok_or("region size overflow")?;
        }
        self.rows(v, prefix, 0, v.offset, contiguous, mode)
    }
    fn rows(
        &mut self,
        v: &View,
        prefix: usize,
        axis: usize,
        offset: u64,
        len: u64,
        mode: Mode,
    ) -> Result<(), String> {
        self.tick()?;
        if axis == prefix {
            return self.span(v, offset, len, mode);
        }
        for i in 0..v.shape[axis] {
            let off = offset
                .checked_add(
                    i.checked_mul(v.strides[axis])
                        .ok_or("region offset overflow")?,
                )
                .ok_or("region offset overflow")?;
            self.rows(v, prefix, axis + 1, off, len, mode)?;
        }
        Ok(())
    }
    fn span(&mut self, v: &View, start: u64, len: u64, mode: Mode) -> Result<(), String> {
        let end = start.checked_add(len).ok_or("region extent overflow")?;
        match &v.elem {
            Elem::Dtype(d) => self.plane(&v.parameter, "", start, end, d.bytes() as u64, mode),
            Elem::Repr(name) => {
                let r =
                    repr::lookup(name).ok_or_else(|| format!("unknown representation `{name}`"))?;
                let cpw = r.codes_per_word() as u64;
                let group = r.group as u64;
                self.plane(
                    &v.parameter,
                    "words",
                    start / cpw,
                    end.div_ceil(cpw),
                    4,
                    mode,
                )?;
                self.plane(
                    &v.parameter,
                    "scale",
                    start / group,
                    end.div_ceil(group),
                    r.coefficient.bytes() as u64,
                    mode,
                )?;
                if r.has_bias {
                    self.plane(
                        &v.parameter,
                        "bias",
                        start / group,
                        end.div_ceil(group),
                        r.coefficient.bytes() as u64,
                        mode,
                    )?;
                }
                Ok(())
            }
            Elem::Param(name) => Err(format!("unbound representation `{name}`")),
        }
    }
    fn plane(
        &mut self,
        param: &str,
        plane: &str,
        start: u64,
        end: u64,
        width: u64,
        mode: Mode,
    ) -> Result<(), String> {
        let default = Backing {
            identity: if plane.is_empty() {
                param.into()
            } else {
                format!("{param}.{plane}")
            },
            byte_offset: 0,
        };
        let backing = self
            .bindings
            .get(&(param.into(), plane.into()))
            .unwrap_or(&default);
        let start = backing
            .byte_offset
            .checked_add(start.checked_mul(width).ok_or("byte offset overflow")?)
            .ok_or("byte offset overflow")?;
        let end = backing
            .byte_offset
            .checked_add(end.checked_mul(width).ok_or("byte extent overflow")?)
            .ok_or("byte extent overflow")?;
        let access = self.account.accesses.backing(&backing.identity);
        if matches!(mode, Mode::Read | Mode::ReadWrite) {
            access.reads.insert(start..end)?;
        }
        if matches!(mode, Mode::Write | Mode::ReadWrite) {
            access.writes.insert(start..end)?;
        }
        Ok(())
    }
}
