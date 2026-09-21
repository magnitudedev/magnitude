//! Calls: casts, toolchain operations, capability intrinsics, and calls of
//! contract families with one candidate binding per definition that unifies
//! with the arguments.

use super::resolve::Sig;
use super::{Checker, LocalKind, ValueClass};
use crate::intrinsics::{
    self, primitive, reduction_result, CapabilitySignature, MathOp, PrimitiveId,
};
use crate::sir::ParamOwnership;
use crate::sir::{
    CandidateBinding, CheckedCall, CheckedExpr, CheckedExprKind, DefId, DefKind, LocalId,
};
use crate::span::Span;
use crate::sym::{Atom, Sym};
use crate::syntax::ast::{self, ExprKind as A};
use crate::types::{DType, Elem, ExtentExpr, TensorType, ValueType};
use std::collections::HashMap;

fn math_of(name: &str) -> Option<(MathOp, usize)> {
    Some(match name {
        "fma" => (MathOp::Fma, 3),
        "exp" => (MathOp::Exp, 1),
        "exp_fast" => (MathOp::ExpFast, 1),
        "rsqrt" => (MathOp::Rsqrt, 1),
        "sqrt" => (MathOp::Sqrt, 1),
        "log" => (MathOp::Log, 1),
        "sin" => (MathOp::Sin, 1),
        "cos" => (MathOp::Cos, 1),
        "abs" => (MathOp::Abs, 1),
        "max" => (MathOp::Max, 2),
        "min" => (MathOp::Min, 2),
        _ => return None,
    })
}

fn single_param(s: &Sym) -> Option<String> {
    let params = s.params();
    match params.as_slice() {
        [p] if *s == Sym::param(p) => Some(p.clone()),
        _ => None,
    }
}

/// Simultaneous substitution of parameter atoms, also inside quotients and remainders.
fn substitute_all(s: &Sym, env: &dyn Fn(&str) -> Option<Sym>) -> Sym {
    let mut out = Sym::constant(0);
    for (monomial, coefficient) in s.monomials() {
        let mut term = Sym::constant(coefficient);
        for (atom, power) in monomial {
            let factor = match atom {
                Atom::Param(p) => env(p).unwrap_or_else(|| Sym::atom(atom.clone())),
                Atom::Quot(n, d) => substitute_all(n, env).quot(&substitute_all(d, env)),
                Atom::Rem(n, d) => substitute_all(n, env).rem(&substitute_all(d, env)),
            };
            for _ in 0..*power {
                term = term.mul(&factor);
            }
        }
        out = out.add(&term);
    }
    out
}

/// Shape and element arguments of one candidate under construction.
#[derive(Default)]
struct Binding {
    shapes: HashMap<String, Sym>,
    elems: HashMap<String, Elem>,
    /// Caller element parameters this candidate requires to be a concrete element.
    requires: Vec<(String, Elem)>,
}

impl<'a> Checker<'a> {
    pub fn call(
        &mut self,
        callee: &ast::Expr,
        bindings: &[(ast::Ident, ast::Expr)],
        args: &[ast::Arg],
        expected: Option<&ValueType>,
        span: Span,
    ) -> Option<CheckedExpr> {
        let name = match &callee.kind {
            A::Name(name) => name,
            A::Attr { base, name } => {
                let (backend, capability) = match &base.kind {
                    A::Attr {
                        base: root,
                        name: capability,
                    } => match &root.kind {
                        A::Name(backend) if self.lookup(&backend.name).is_none() => {
                            (backend, capability)
                        }
                        _ => {
                            self.error(
                                callee.span,
                                "capability intrinsics use `<backend>.<capability>.<operation>`",
                            );
                            return None;
                        }
                    },
                    _ => {
                        self.error(
                            callee.span,
                            "capability intrinsics use `<backend>.<capability>.<operation>`",
                        );
                        return None;
                    }
                };
                return self.intrinsic(backend, capability, name, args, span);
            }
            _ => {
                self.error(callee.span, "only named functions can be called; functions and operations resolve statically");
                return None;
            }
        };
        if self.lookup(&name.name).is_some() {
            self.error(
                name.span,
                format!("`{}` is a value, not a function", name.name),
            );
            return None;
        }
        let toolchain = DType::from_name(&name.name).is_some()
            || math_of(&name.name).is_some()
            || matches!(
                name.name.as_str(),
                "load"
                    | "to_owned"
                    | "clone"
                    | "decode"
                    | "zeros_like"
                    | "ones_like"
                    | "select"
                    | "reduce"
                    | "reshape"
                    | "extent"
                    | "coord"
                    | "capacity"
                    | "valid"
                    | "atomic"
                    | "owned"
                    | "axis"
                    | "lanes"
                    | "full"
            );
        if toolchain && !bindings.is_empty() {
            self.error(
                span,
                format!(
                    "`{}` is a toolchain operation and takes no shape bindings",
                    name.name
                ),
            );
            return None;
        }
        if let Some(dtype) = DType::from_name(&name.name) {
            return self.cast(dtype, args, span);
        }
        if let Some((op, arity)) = math_of(&name.name) {
            return self.math(op, arity, &name.name, args, expected, span);
        }
        let positional = |c: &mut Checker, n: usize| -> bool {
            let ok = args.len() == n && args.iter().all(|a| a.name.is_none());
            if !ok {
                c.error(
                    span,
                    format!("`{}` takes {n} positional argument(s)", name.name),
                );
            }
            ok
        };
        match name.name.as_str() {
            "to_owned" => {
                if !positional(self, 1) {
                    return None;
                }
                let value = self.expr(&args[0].value, None)?;
                // Ownership conversion is idempotent. An already-owned value
                // needs neither a diagnostic nor a second allocation.
                if matches!(self.class_of(&value), ValueClass::Owned) {
                    return Some(value);
                }
                if !matches!(
                    self.class_of(&value),
                    ValueClass::Borrowed | ValueClass::Computed
                ) {
                    self.error(
                        value.span,
                        format!(
                            "`to_owned` materializes a borrowed or computed tensor value, found {}",
                            value.ty
                        ),
                    );
                    return None;
                }
                if !primitive(PrimitiveId::Materialize).accepts(&[value.ty.clone()]) {
                    self.error(
                        value.span,
                        format!("`to_owned` is not defined on {}", value.ty),
                    );
                    return None;
                }
                let ty = value.ty.clone();
                Some(CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id: PrimitiveId::Materialize,
                        operands: vec![value],
                    },
                    ty,
                    None,
                    span,
                ))
            }
            "clone" => {
                if !positional(self, 1) {
                    return None;
                }
                let value = self.expr(&args[0].value, None)?;
                if !matches!(self.class_of(&value), ValueClass::Owned) {
                    self.error(
                        value.span,
                        format!("`clone` duplicates an owned tensor, found {}", value.ty),
                    );
                    return None;
                }
                if !primitive(PrimitiveId::Clone).accepts(&[value.ty.clone()]) {
                    self.error(
                        value.span,
                        format!("`clone` is not defined on {}", value.ty),
                    );
                    return None;
                }
                let ty = value.ty.clone();
                Some(CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id: PrimitiveId::Clone,
                        operands: vec![value],
                    },
                    ty,
                    None,
                    span,
                ))
            }
            "load" => {
                if !positional(self, 1) {
                    return None;
                }
                let v = self.expr(&args[0].value, None)?;
                if !matches!(self.class_of(&v), ValueClass::Borrowed | ValueClass::Owned) {
                    self.error(
                        v.span,
                        format!(
                            "`load` snapshots borrowed or owned storage in its own representation, found {}",
                            v.ty
                        ),
                    );
                    return None;
                }
                if !primitive(PrimitiveId::Load).accepts(&[v.ty.clone()]) {
                    self.error(v.span, format!("`load` is not defined on {}", v.ty));
                    return None;
                }
                let ty = v.ty.clone();
                Some(CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id: PrimitiveId::Load,
                        operands: vec![v],
                    },
                    ty,
                    None,
                    span,
                ))
            }
            "decode" => {
                if !positional(self, 1) {
                    return None;
                }
                let v = self.expr(&args[0].value, None)?;
                let Some(s) = v.ty.shaped().filter(|s| !matches!(s.elem, Elem::Dtype(_))) else {
                    self.error(v.span, format!("`decode` produces the dense `f32` value of a packed view, found {}; convert dense values with a cast", v.ty));
                    return None;
                };
                let ty =
                    ValueType::Tensor(TensorType::new(s.axes.clone(), Elem::Dtype(DType::F32)));
                Some(CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id: PrimitiveId::Decode,
                        operands: vec![v],
                    },
                    ty,
                    None,
                    span,
                ))
            }
            "zeros_like" | "ones_like" => {
                let (Some(like), dtype) = (args.first().filter(|a| a.name.is_none()), args.get(1))
                else {
                    self.error(
                        span,
                        format!(
                            "`{}(v, dtype=f32)` takes a shaped value and an optional dtype",
                            name.name
                        ),
                    );
                    return None;
                };
                if args.len() > 2 {
                    self.error(
                        span,
                        format!(
                            "`{}(v, dtype=f32)` takes a shaped value and an optional dtype",
                            name.name
                        ),
                    );
                    return None;
                }
                // Only the shape is used; the value is not read.
                let like = self.expr_inner(&like.value, None, true)?;
                let Some(s) = like.ty.shaped().cloned() else {
                    self.error(
                        like.span,
                        format!(
                            "`{}` takes the shape of a tensor, view or tile, found {}",
                            name.name, like.ty
                        ),
                    );
                    return None;
                };
                let dtype = match dtype {
                    Some(ast::Arg {
                        name: Some(label),
                        value,
                    }) if label.name == "dtype" => match &value.kind {
                        A::Name(n) => DType::from_name(&n.name),
                        _ => None,
                    },
                    Some(_) => None,
                    None => Some(s.elem.read_dtype().unwrap_or(DType::F32)),
                };
                let Some(dtype) = dtype else {
                    self.error(span, "the second argument is `dtype=<dtype name>`");
                    return None;
                };
                let ty = ValueType::Tensor(TensorType::new(s.axes, Elem::Dtype(dtype)));
                let value = if name.name == "zeros_like" { 0.0 } else { 1.0 };
                let id = PrimitiveId::Fill { value, dtype };
                Some(CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id,
                        operands: vec![like],
                    },
                    ty,
                    None,
                    span,
                ))
            }
            "select" => {
                if !positional(self, 3) {
                    return None;
                }
                let cond = self.expr(&args[0].value, None)?;
                let then = self.expr(&args[1].value, expected)?;
                let els = self.expr(&args[2].value, Some(&then.ty))?;
                let (axes, dtypes) = self.broadcast(&[&cond, &then, &els], "`select`", span)?;
                if dtypes[0] != DType::Bool {
                    self.error(
                        cond.span,
                        format!(
                            "the first operand of `select` is a mask (`bool`), found {}",
                            dtypes[0].name()
                        ),
                    );
                    return None;
                }
                if DType::promote(dtypes[1], dtypes[2]).is_none() {
                    self.error(
                        span,
                        format!(
                            "`select` between {} and {} needs an explicit cast",
                            dtypes[1].name(),
                            dtypes[2].name()
                        ),
                    );
                    return None;
                }
                self.elementwise_primitive(PrimitiveId::Select, vec![cond, then, els], axes, span)
            }
            "reduce" => {
                let unordered = match args.get(3) {
                    Some(ast::Arg {
                        name: Some(label),
                        value:
                            ast::Expr {
                                kind: A::Bool(value),
                                ..
                            },
                    }) if label.name == "unordered" && args.len() == 4 => Some(*value),
                    None => Some(false),
                    _ => None,
                };
                let Some(unordered) = unordered
                    .filter(|_| args.len() >= 3 && args[..3].iter().all(|a| a.name.is_none()))
                else {
                    self.error(span, "`reduce(tile, axis, op)` takes three positional arguments and an optional `unordered=true|false`");
                    return None;
                };
                self.reduce(args, unordered, span)
            }
            "reshape" => {
                if !positional(self, 2) {
                    return None;
                }
                self.reshape(args, span)
            }
            "extent" | "valid" => {
                if !positional(self, 2) {
                    return None;
                }
                if name.name == "valid" && !self.target_form(span, "`valid`", None) {
                    return None;
                }
                let base = self.expr_inner(&args[0].value, None, true)?;
                let Some(s) = base.ty.shaped().cloned() else {
                    self.error(
                        base.span,
                        format!(
                            "`{}` needs a tensor, view or tile, found {}",
                            name.name, base.ty
                        ),
                    );
                    return None;
                };
                let axis = self.constant_axis(&args[1].value, s.rank())?;
                let Some(sym) = s.axes[axis].sym().cloned() else {
                    self.error(span, "axis extents are symbolic at checked scope");
                    return None;
                };
                self.numeric_use(&sym);
                let id = if name.name == "extent" {
                    PrimitiveId::Extent { axis }
                } else {
                    PrimitiveId::ValidExtent { axis }
                };
                Some(self.scalar_expr(
                    CheckedExprKind::Primitive {
                        id,
                        operands: vec![base],
                    },
                    DType::I32,
                    Some(sym),
                    span,
                ))
            }
            "capacity" => {
                self.error(
                    span,
                    "`capacity` was removed: capacity is a planning resource bound, never a source value",
                );
                None
            }
            "coord" => {
                self.error(
                    span,
                    "`coord(i)` was removed with tile coordinates; iterate a bounded range",
                );
                None
            }
            "atomic" => self.atomic(args, span),
            "owned" | "axis" | "lanes" => {
                self.error(
                    span,
                    format!("`{}` is an iterator and appears only in `for`", name.name),
                );
                None
            }
            "full" => {
                self.error(span, "`full(X)` was removed with structural slices");
                None
            }
            _ => self.user_call(name, bindings, args, span),
        }
    }

    fn constant_axis(&mut self, e: &ast::Expr, rank: usize) -> Option<usize> {
        let axis = match &e.kind {
            A::Int(v) => usize::try_from(*v).ok().filter(|a| *a < rank),
            _ => None,
        };
        if axis.is_none() {
            self.error(e.span, format!("the axis is a constant below rank {rank}"));
        }
        axis
    }

    fn math(
        &mut self,
        op: MathOp,
        arity: usize,
        name: &str,
        args: &[ast::Arg],
        expected: Option<&ValueType>,
        span: Span,
    ) -> Option<CheckedExpr> {
        if args.len() != arity || args.iter().any(|a| a.name.is_some()) {
            self.error(
                span,
                format!("`{name}` takes {arity} positional argument(s)"),
            );
            return None;
        }
        let float_only = !op.numeric_operands();
        let default = ValueType::Scalar(DType::F32);
        let mut hint: Option<ValueType> = expected
            .cloned()
            .or_else(|| float_only.then(|| default.clone()));
        let mut out: Vec<CheckedExpr> = Vec::new();
        for arg in args {
            let e = self.expr(&arg.value, hint.as_ref())?;
            if out.is_empty() || matches!(args[0].value.kind, A::Int(_) | A::Float(_)) {
                hint = Some(e.ty.clone());
            }
            out.push(e);
        }
        // A leading literal adopts the dtype of the operand it combines with.
        if out.len() > 1 && matches!(args[0].value.kind, A::Int(_) | A::Float(_)) {
            out[0] = self.expr(&args[0].value, Some(&out[1].ty.clone()))?;
        }
        let operands: Vec<&CheckedExpr> = out.iter().collect();
        let (axes, dtypes) = self.broadcast(&operands, &format!("`{name}`"), span)?;
        let mut dtype = dtypes[0];
        for d in &dtypes[1..] {
            match DType::promote(dtype, *d) {
                Some(p) => dtype = p,
                None => {
                    self.error(
                        span,
                        format!(
                            "`{name}` between {} and {} needs an explicit cast",
                            dtype.name(),
                            d.name()
                        ),
                    );
                    return None;
                }
            }
        }
        if if float_only {
            !dtype.is_float()
        } else {
            !dtype.is_numeric()
        } {
            self.error(
                span,
                format!(
                    "`{name}` needs {} operands, found {}",
                    if float_only { "float" } else { "numeric" },
                    dtype.name()
                ),
            );
            return None;
        }
        self.elementwise_primitive(PrimitiveId::Math(op), out, axes, span)
    }

    fn reduce(&mut self, args: &[ast::Arg], unordered: bool, span: Span) -> Option<CheckedExpr> {
        let t = self.expr(&args[0].value, None)?;
        let Some(s) = t.ty.shaped() else {
            self.error(
                t.span,
                format!(
                    "`reduce` reduces a dense tile value, found {}; read storage with `load` or a cast",
                    t.ty
                ),
            );
            return None;
        };
        // Storage realization is not part of reduction syntax. A computed
        // operand is a legitimate value without logical storage.
        let Some(dtype) = (match &s.elem {
            Elem::Dtype(d) => Some(*d),
            Elem::Param(_) => Some(DType::F32),
            Elem::Repr(_) => None,
        }) else {
            self.error(
                t.span,
                "`reduce` reduces a dense tile; decode packed values first",
            );
            return None;
        };
        let axis = self.constant_axis(&args[1].value, s.rank())?;
        let op = match &args[2].value.kind {
            A::Name(n) => match n.name.as_str() {
                "sum" => Some(intrinsics::ReduceOp::Sum),
                "max" => Some(intrinsics::ReduceOp::Max),
                "min" => Some(intrinsics::ReduceOp::Min),
                "argmax" => Some(intrinsics::ReduceOp::Argmax),
                _ => None,
            },
            _ => None,
        };
        let Some(op) = op else {
            self.error(
                args[2].value.span,
                "`reduce` needs an operation: sum, max, min or argmax",
            );
            return None;
        };
        if unordered && op == intrinsics::ReduceOp::Argmax {
            self.error(span, "`argmax` has one defined winner (ties go to the smaller index) and never accepts `unordered`");
            return None;
        }
        if let Some(ExtentExpr::Sym(extent)) = s.axes.get(axis) {
            if let Some(p) = single_param(extent).filter(|p| self.sig.shape_params.contains(p)) {
                self.summary.reduces.insert(p);
            }
        }
        let _ = dtype;
        let id = PrimitiveId::Reduce {
            op,
            axis,
            unordered,
        };
        let ty = reduction_result(&t.ty, op, axis)
            .ok_or_else(|| {
                self.error(
                    t.span,
                    "`reduce` reduces a dense tile; decode packed values first",
                );
            })
            .ok()?;
        Some(CheckedExpr::new(
            CheckedExprKind::Primitive {
                id,
                operands: vec![t],
            },
            ty,
            None,
            span,
        ))
    }

    fn reshape(&mut self, args: &[ast::Arg], span: Span) -> Option<CheckedExpr> {
        let base = self.expr_inner(&args[0].value, None, true)?;
        let Some(s) = base.ty.shaped().cloned() else {
            self.error(
                base.span,
                format!("`reshape` needs a tensor, view or tile, found {}", base.ty),
            );
            return None;
        };
        if matches!(s.elem, Elem::Repr(_)) {
            self.error(span, "`reshape` requires dense storage: packets run along the last axis of a packed value");
            return None;
        }
        let A::Tuple(dimensions) = &args[1].value.kind else {
            self.error(
                args[1].value.span,
                "the `reshape` target is a tuple of shape expressions",
            );
            return None;
        };
        let mut source = Sym::constant(1);
        for axis in &s.axes {
            // A static extent is the degenerate symbolic product: a constant.
            let extent = match axis {
                ExtentExpr::Static(value) => Sym::constant(*value as i64),
                ExtentExpr::Sym(extent) => extent.clone(),
                ExtentExpr::Runtime(_) => {
                    self.error(span, "`reshape` needs symbolic source extents");
                    return None;
                }
            };
            source = source.mul(&extent);
        }
        let mut axes = Vec::new();
        let mut operands = vec![base];
        let mut target = Sym::constant(1);
        for dimension in dimensions {
            let d = self.expr(dimension, Some(&ValueType::Scalar(DType::I32)))?;
            let Some(extent) = d.sym.clone() else {
                self.error(d.span, "`reshape` extents are symbolic integer expressions");
                return None;
            };
            self.require_nonneg(&extent, d.span, "reshape extent may be negative");
            self.numeric_use(&extent);
            target = target.mul(&extent);
            axes.push(crate::sir::sym_extent(extent));
            operands.push(d);
        }
        if axes.is_empty() || !self.prover().zero(&target.sub(&source)) {
            self.error(span, format!("`reshape` must provably preserve element correspondence: `{source}` elements into `{target}`"));
            return None;
        }
        let shaped = TensorType::new(axes, s.elem);
        let ty = ValueType::Tensor(shaped);
        Some(CheckedExpr::new(
            CheckedExprKind::Primitive {
                id: PrimitiveId::Reshape,
                operands,
            },
            ty,
            None,
            span,
        ))
    }

    /// `atomic(add|max|min, place, value)`: one registry primitive with an
    /// atomic effect, portable in every body. Defined for f32/f16/bf16/i32/u32
    /// elements; bool is rejected because bool arithmetic is undefined, and
    /// packed planes are not writable.
    fn atomic(&mut self, args: &[ast::Arg], span: Span) -> Option<CheckedExpr> {
        if !positional_3(args) {
            self.error(
                span,
                "`atomic(op, place, value)` takes three positional arguments",
            );
            return None;
        }
        let op = match &args[0].value.kind {
            A::Name(n) => match intrinsics::AtomicOp::parse(&n.name) {
                Some(op) => op,
                None => {
                    self.error(
                        n.span,
                        format!("`atomic({}, …)`: the operation is add, max or min", n.name),
                    );
                    return None;
                }
            },
            _ => {
                self.error(
                    args[0].value.span,
                    "`atomic` needs an operation name: add, max or min",
                );
                return None;
            }
        };
        let (root, indices, selected) = match &args[1].value.kind {
            A::Name(name) => {
                let Some(id) = self.lookup(&name.name) else {
                    self.error(name.span, format!("`{}` is not declared", name.name));
                    return None;
                };
                (id, Vec::new(), self.locals[id].ty.clone())
            }
            A::Index { .. } => self.place(&args[1].value)?,
            _ => {
                self.error(
                    args[1].value.span,
                    "the `atomic` place is a tensor element: `atomic(add, t[i], v)`",
                );
                return None;
            }
        };
        let ValueType::Scalar(dtype) = selected else {
            self.error(
                args[1].value.span,
                format!("the `atomic` place selects one element, found {}", selected),
            );
            return None;
        };
        if !intrinsics::atomic_dtype(dtype) {
            self.error(
                span,
                format!(
                    "`atomic(add, …)` is defined for f32, f16, bf16, i32 and u32 elements, not {}",
                    dtype.name()
                ),
            );
            return None;
        }
        // Packed planes are readable but not writable.
        if let Some(shaped) = self.locals[root].ty.shaped() {
            if matches!(shaped.elem, Elem::Repr(_)) {
                self.error(
                    span,
                    "packed representations are readable and decodable but not writable",
                );
                return None;
            }
        }
        let value = self.expr(&args[2].value, Some(&ValueType::Scalar(dtype)))?;
        if value.ty.scalar_dtype() != Some(dtype) {
            self.error(
                value.span,
                format!(
                    "the `atomic` value must be {}, found {}",
                    dtype.name(),
                    value.ty
                ),
            );
            return None;
        }
        let binding = match &args[1].value.kind {
            A::Name(name) => self.lookup(&name.name),
            A::Index { base, .. } => match &base.kind {
                A::Name(name) => self.lookup(&name.name),
                _ => None,
            },
            _ => None,
        };
        let Some(binding) = binding else {
            self.error(span, "the `atomic` place does not name a binding");
            return None;
        };
        if !self.writable_root(self.root_var_local(binding)) && !self.writable_root(root) {
            self.error(
                span,
                "an `atomic` place is a `&mut tensor` parameter or local `let mut` state",
            );
            return None;
        }
        if self
            .borrows
            .iter()
            .any(|(borrow, (borrowed, _))| *borrowed == root && *borrow != binding)
        {
            self.error(
                span,
                format!(
                    "cannot update `{}` atomically while a tensor borrow is live",
                    self.locals[root].name
                ),
            );
            return None;
        }
        // Atomics are the one admitted cross-visit update of an independent
        // loop; record them for the mutation summary.
        let storage_root = self.root_var_local(binding);
        for ctx in self.loops.iter_mut() {
            if storage_root < ctx.floor {
                ctx.atomics.push(storage_root);
            }
        }
        self.mutated.push(storage_root);
        let arity = indices.len();
        let mut operands = vec![CheckedExpr::new(
            CheckedExprKind::Local(binding),
            self.locals[binding].ty.clone(),
            None,
            span,
        )];
        for index in &indices {
            match index {
                crate::sir::CheckedIndex::Point(p) => operands.push(p.clone()),
                crate::sir::CheckedIndex::Range { start, end } => {
                    operands.extend(start.iter().chain(end).cloned());
                }
            }
        }
        operands.push(value);
        Some(CheckedExpr::new(
            CheckedExprKind::Primitive {
                id: PrimitiveId::Atomic { op, arity },
                operands,
            },
            ValueType::Void,
            None,
            span,
        ))
    }

    /// The storage root of a binding local, before following view aliases.
    pub(crate) fn root_var_local(&self, id: LocalId) -> LocalId {
        self.view_roots.get(&id).copied().unwrap_or(id)
    }

    // ---- capability intrinsics ----

    fn intrinsic(
        &mut self,
        backend: &ast::Ident,
        capability: &ast::Ident,
        name: &ast::Ident,
        args: &[ast::Arg],
        span: Span,
    ) -> Option<CheckedExpr> {
        if !intrinsics::known_backend(&backend.name) {
            self.error(
                backend.span,
                format!("`{}` is not a value or a backend namespace", backend.name),
            );
            return None;
        }
        let Some(capability_id) = intrinsics::capability(&backend.name, &capability.name) else {
            self.error(
                capability.span,
                format!(
                    "`{}.{}` is not a known capability namespace",
                    backend.name, capability.name
                ),
            );
            return None;
        };
        if !self.target_form(
            span,
            &format!("`{}.{}.{}`", backend.name, capability.name, name.name),
            Some(&backend.name),
        ) {
            return None;
        }
        let signatures = intrinsics::lookup(&backend.name, &capability.name, &name.name);
        if signatures.is_empty() {
            self.error(
                name.span,
                format!(
                    "capability `{}` has no intrinsic `{}`",
                    capability_id.path(),
                    name.name
                ),
            );
            return None;
        }
        self.use_capability(
            &capability_id,
            span,
            &format!(
                "intrinsic `{}.{}.{}`",
                backend.name, capability.name, name.name
            ),
        );
        self.intrinsic_call(&signatures, args, span)
    }

    fn intrinsic_call(
        &mut self,
        signatures: &[CapabilitySignature],
        args: &[ast::Arg],
        span: Span,
    ) -> Option<CheckedExpr> {
        // Logical matrix intrinsics carry an `accumulation=` named argument.
        if matches!(
            signatures.first().map(|s| &s.semantics),
            Some(
                intrinsics::CapabilitySemantics::MatrixMatmul
                    | intrinsics::CapabilitySemantics::MatrixMatmulAdd
            )
        ) {
            return self.matrix_intrinsic(signatures, args, span);
        }
        let mut checked = Vec::new();
        for arg in args {
            if arg.name.is_some() {
                self.error(
                    arg.value.span,
                    format!(
                        "`{}` takes positional arguments only",
                        signatures[0].id.path()
                    ),
                );
                return None;
            }
            let e = self.expr(&arg.value, None)?;
            checked.push(e);
        }
        let matching: Vec<&CapabilitySignature> = signatures
            .iter()
            .filter(|s| {
                s.arguments.len() == checked.len()
                    && s.arguments
                        .iter()
                        .zip(&checked)
                        .all(|(p, a)| capability_argument_matches(p, &a.ty))
            })
            .collect();
        let [signature] = matching.as_slice() else {
            self.error(
                span,
                format!(
                    "`{}` takes {} argument(s) of its declared types; {} given",
                    signatures[0].id.path(),
                    signatures.first().map(|s| s.arguments.len()).unwrap_or(0),
                    checked.len()
                ),
            );
            return None;
        };
        let id = signature.id.clone();
        let result = signature.result.clone();
        let used = crate::sir::IntrinsicUse {
            id: id.clone(),
            arguments: checked.iter().map(|a| a.ty.clone()).collect(),
            result: result.clone(),
        };
        if !self.intrinsic_uses.contains(&used) {
            self.intrinsic_uses.push(used);
        }
        Some(CheckedExpr::new(
            CheckedExprKind::Capability { id, args: checked },
            result,
            None,
            span,
        ))
    }

    fn matrix_intrinsic(
        &mut self,
        signatures: &[CapabilitySignature],
        args: &[ast::Arg],
        span: Span,
    ) -> Option<CheckedExpr> {
        let matmul = matches!(
            signatures[0].semantics,
            intrinsics::CapabilitySemantics::MatrixMatmul
        );
        let mut positional = Vec::new();
        let mut accumulation: Option<DType> = None;
        for argument in args {
            match argument.name.as_ref().map(|name| name.name.as_str()) {
                None => positional.push(&argument.value),
                Some("accumulation") if matmul => {
                    let dtype = match &argument.value.kind {
                        A::Name(name) => DType::from_name(&name.name),
                        _ => None,
                    };
                    let Some(dtype) = dtype.filter(|dtype| dtype.is_numeric()) else {
                        self.error(
                            argument.value.span,
                            "matrix accumulation must name a numeric dtype",
                        );
                        return None;
                    };
                    if accumulation.replace(dtype).is_some() {
                        self.error(
                            argument.value.span,
                            "`accumulation` is supplied more than once",
                        );
                        return None;
                    }
                }
                Some(name) => {
                    self.error(
                        argument.value.span,
                        format!(
                            "`{}` has no named argument `{name}`",
                            signatures[0].id.path()
                        ),
                    );
                    return None;
                }
            }
        }
        let expected = if matmul { 2 } else { 3 };
        if positional.len() != expected {
            self.error(
                span,
                format!(
                    "`{}` takes {expected} positional argument(s)",
                    signatures[0].id.path()
                ),
            );
            return None;
        }
        if matmul && accumulation.is_none() {
            self.error(
                span,
                "`matrix.matmul` requires the named argument `accumulation=<dtype>`",
            );
            return None;
        }
        let mut checked = Vec::new();
        for operand in positional {
            let expression = self.expr(operand, None)?;
            if !matches!(&expression.ty, ValueType::Tensor(shape) if shape.rank() == 2) {
                self.error(
                    expression.span,
                    format!(
                        "`{}` needs rank-two logical tensor operands, found {}",
                        signatures[0].id.path(),
                        expression.ty
                    ),
                );
                return None;
            }
            checked.push(expression);
        }
        let left = checked[0].ty.shaped().expect("checked rank-two operand");
        let right = checked[1].ty.shaped().expect("checked rank-two operand");
        if !self.same_extent(&left.axes[1], &right.axes[0]) {
            self.error(
                span,
                format!(
                    "`{}` inner axes differ: {} versus {}",
                    signatures[0].id.path(),
                    left.axes[1],
                    right.axes[0]
                ),
            );
            return None;
        }
        let (element, result_axes) = if matmul {
            (
                Elem::Dtype(accumulation.expect("checked above")),
                vec![left.axes[0].clone(), right.axes[1].clone()],
            )
        } else {
            let accumulator = checked[2]
                .ty
                .shaped()
                .expect("checked rank-two accumulator");
            let expected_axes = [&left.axes[0], &right.axes[1]];
            if accumulator
                .axes
                .iter()
                .zip(expected_axes)
                .any(|(actual, expected)| !self.same_extent(actual, expected))
            {
                self.error(
                    checked[2].span,
                    format!(
                        "`{}` accumulator shape does not match the matrix product",
                        signatures[0].id.path()
                    ),
                );
                return None;
            }
            (accumulator.elem.clone(), accumulator.axes.clone())
        };
        let id = signatures[0].id.clone();
        let result = ValueType::Tensor(TensorType::new(result_axes, element));
        let used = crate::sir::IntrinsicUse {
            id: id.clone(),
            arguments: checked.iter().map(|a| a.ty.clone()).collect(),
            result: result.clone(),
        };
        if !self.intrinsic_uses.contains(&used) {
            self.intrinsic_uses.push(used);
        }
        Some(CheckedExpr::new(
            CheckedExprKind::Capability { id, args: checked },
            result,
            None,
            span,
        ))
    }

    // ---- contract families ----

    /// Parameter ordinal -> argument ordinal. `into=` binds the one remaining
    /// `inout` parameter when no parameter is named `into`.
    fn arg_order(sig: &Sig, args: &[ast::Arg]) -> Result<Vec<usize>, String> {
        if args.len() != sig.params.len() {
            return Err(format!(
                "takes {} arguments, {} given",
                sig.params.len(),
                args.len()
            ));
        }
        let mut order: Vec<Option<usize>> = vec![None; sig.params.len()];
        let mut positional = 0;
        let mut into = None;
        for (ordinal, arg) in args.iter().enumerate() {
            let slot = match &arg.name {
                None => {
                    positional += 1;
                    positional - 1
                }
                Some(label) => match sig.params.iter().position(|p| p.name == label.name) {
                    Some(slot) => slot,
                    None if label.name == "into" && into.is_none() => {
                        into = Some(ordinal);
                        continue;
                    }
                    None => return Err(format!("has no parameter `{}`", label.name)),
                },
            };
            if order.get(slot).is_none_or(|bound| bound.is_some()) {
                return Err(format!("binds parameter {} twice", slot + 1));
            }
            order[slot] = Some(ordinal);
        }
        if let Some(ordinal) = into {
            let open: Vec<usize> = (0..order.len())
                .filter(|i| order[*i].is_none() && sig.params[*i].mode != crate::sir::Mode::In)
                .collect();
            let [slot] = open.as_slice() else {
                return Err("`into=` binds the one unbound `inout` parameter".to_string());
            };
            order[*slot] = Some(ordinal);
        }
        order
            .into_iter()
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| "leaves a parameter unbound".to_string())
    }

    /// Substitute a candidate's bound shape parameters into one symbolic extent.
    fn substitute(s: &Sym, binding: &Binding) -> Result<Sym, String> {
        for p in s.params() {
            if !binding.shapes.contains_key(&p) {
                return Err(format!(
                    "shape parameter `{p}` is not determined by the arguments"
                ));
            }
        }
        Ok(substitute_all(s, &|p| binding.shapes.get(p).cloned()))
    }

    fn substitute_ty(ty: &ValueType, binding: &Binding) -> Result<ValueType, String> {
        let shaped = |s: &TensorType| -> Result<TensorType, String> {
            let mut axes = Vec::new();
            for axis in &s.axes {
                axes.push(match axis {
                    ExtentExpr::Sym(sym) => {
                        match single_param(sym).and_then(|p| binding.shapes.get(&p)) {
                            Some(extent) => crate::sir::sym_extent(extent.clone()),
                            None => crate::sir::sym_extent(Self::substitute(sym, binding)?),
                        }
                    }
                    other => other.clone(),
                });
            }
            let elem = match &s.elem {
                Elem::Param(p) => binding.elems.get(p).cloned().ok_or_else(|| {
                    format!("element parameter `{p}` is not determined by the arguments")
                })?,
                other => other.clone(),
            };
            s.specialize_elem(axes, elem)
        };
        Ok(match ty {
            ValueType::Tensor(s) => ValueType::Tensor(shaped(s)?),
            ValueType::Index { bound } => ValueType::Index {
                bound: crate::sir::sym_extent(Self::substitute(
                    bound.sym().ok_or("index bound is not symbolic")?,
                    binding,
                )?),
            },
            ValueType::Range { bound } => ValueType::Range {
                bound: crate::sir::sym_extent(Self::substitute(
                    bound.sym().ok_or("range bound is not symbolic")?,
                    binding,
                )?),
            },
            ValueType::Tuple(items) => ValueType::Tuple(
                crate::types::NonEmpty::new(
                    items
                        .iter()
                        .map(|t| Self::substitute_ty(t, binding))
                        .collect::<Result<Vec<_>, _>>()?,
                )
                .ok_or("empty tuple")?,
            ),
            other => other.clone(),
        })
    }

    fn unify(
        &self,
        param: &ValueType,
        arg: &ValueType,
        arg_class: ValueClass,
        binding: &mut Binding,
        compound: &mut Vec<(Sym, Sym)>,
    ) -> Result<(), String> {
        let mismatch = || Err(format!("expects {param} but was given {arg}"));
        match (param, arg) {
            (ValueType::Scalar(a), _) => match arg.scalar_dtype() {
                Some(b) if *a == b || (a.is_float() && b.is_float()) => Ok(()),
                _ => mismatch(),
            },
            (ValueType::Index { .. }, _) if arg.scalar_dtype() == Some(DType::I32) => Ok(()),
            (ValueType::Range { bound: p }, ValueType::Range { bound: a }) => {
                let p = p.sym().ok_or("range bound is not symbolic")?;
                let a = a.sym().ok_or("range bound is not symbolic")?;
                match single_param(p) {
                    Some(name) => match binding.shapes.get(&name) {
                        Some(bound) if !self.prover().zero(&bound.sub(a)) => mismatch(),
                        Some(_) => Ok(()),
                        None => {
                            binding.shapes.insert(name, a.clone());
                            Ok(())
                        }
                    },
                    None if self.prover().zero(&p.sub(a)) => Ok(()),
                    None => mismatch(),
                }
            }
            (ValueType::Tensor(p), ValueType::Tensor(a)) => {
                // Ownership determines which argument classes a tensor parameter
                // accepts; the type itself is canonical.
                if p.rank() != a.rank() {
                    return mismatch();
                }
                for (pd, ad) in p.axes.iter().zip(&a.axes) {
                    let to_sym = |e: &ExtentExpr| -> Option<Sym> {
                        match e {
                            ExtentExpr::Sym(s) => Some(s.clone()),
                            ExtentExpr::Static(n) => Some(Sym::constant(*n as i64)),
                            ExtentExpr::Runtime(_) => None,
                        }
                    };
                    let (Some(pd), Some(ad)) = (to_sym(pd), to_sym(ad)) else {
                        return mismatch();
                    };
                    match single_param(&pd) {
                        Some(name) => match binding.shapes.get(&name) {
                            Some(bound) if !self.prover().zero(&bound.sub(&ad)) => {
                                return Err(format!(
                                    "binds shape parameter `{name}` to both {bound} and {ad}"
                                ));
                            }
                            Some(_) => {}
                            None => {
                                binding.shapes.insert(name, ad.clone());
                            }
                        },
                        None => compound.push((pd.clone(), ad.clone())),
                    }
                }
                let ok = match (&p.elem, &a.elem) {
                    (Elem::Param(name), actual) => match binding.elems.get(name) {
                        Some(bound) => bound == actual,
                        None => {
                            let admitted = !matches!(actual, Elem::Dtype(d) if !d.is_float());
                            if admitted {
                                binding.elems.insert(name.clone(), actual.clone());
                            }
                            admitted
                        }
                    },
                    // A concrete callee element against a caller element parameter unifies
                    // provisionally: the candidate applies where that parameter is this element.
                    (concrete, Elem::Param(own)) => {
                        match binding.requires.iter().find(|(p, _)| p == own) {
                            Some((_, required)) => required == concrete,
                            None => {
                                binding.requires.push((own.clone(), concrete.clone()));
                                true
                            }
                        }
                    }
                    (x, y) => x == y,
                };
                let packets_aligned =
                    !matches!(a.elem, Elem::Repr(_)) || a.packed_axis == Some(a.rank() - 1);
                if ok && packets_aligned {
                    Ok(())
                } else {
                    mismatch()
                }
            }
            (ValueType::Tuple(p), ValueType::Tuple(a)) if p.len() == a.len() => p
                .iter()
                .zip(a.iter())
                .try_for_each(|(p, a)| self.unify(p, a, arg_class, binding, compound)),
            (ValueType::CapabilityValue(p), ValueType::CapabilityValue(a)) if p == a => Ok(()),
            _ => mismatch(),
        }
    }

    /// Bind one candidate definition to the checked arguments.
    fn bind_candidate(
        &self,
        sig: &Sig,
        explicit: &[(String, Sym)],
        ast_args: &[ast::Arg],
        args: &[CheckedExpr],
    ) -> Result<(Vec<usize>, Binding), String> {
        let order = Self::arg_order(sig, ast_args)?;
        let mut binding = Binding::default();
        for (name, extent) in explicit {
            if !sig.shape_params.contains(name) {
                return Err(format!("has no shape parameter `{name}`"));
            }
            binding.shapes.insert(name.clone(), extent.clone());
        }
        let mut compound = Vec::new();
        for (param, ordinal) in sig.params.iter().zip(&order) {
            let argument = &args[*ordinal];
            let arg_class = self.class_of(argument);
            // Ownership admission before type unification.
            match param.ownership {
                ParamOwnership::Owned
                    if !matches!(arg_class, ValueClass::Owned | ValueClass::Borrowed) =>
                {
                    return Err(format!(
                        "parameter `{}` takes an owned tensor or a tensor place, found a computed value",
                        param.name
                    ));
                }
                ParamOwnership::Exclusive
                    if !matches!(arg_class, ValueClass::Owned | ValueClass::Borrowed) =>
                {
                    return Err(format!(
                        "parameter `{}` requires a tensor place for exclusive access",
                        param.name
                    ));
                }
                _ => {}
            }
            self.unify(
                &param.ty,
                &argument.ty,
                arg_class,
                &mut binding,
                &mut compound,
            )
            .map_err(|e| format!("parameter `{}` {e}", param.name))?;
        }
        // Compound extents with one unknown, linear in it, determine that parameter.
        loop {
            let mut changed = false;
            for (pd, ad) in &compound {
                let unknown: Vec<String> = pd
                    .params()
                    .into_iter()
                    .filter(|p| !binding.shapes.contains_key(p))
                    .collect();
                let [u] = unknown.as_slice() else { continue };
                let hole = Atom::Param(format!("?{u}"));
                let mut trial = Binding {
                    shapes: binding.shapes.clone(),
                    ..Binding::default()
                };
                trial.shapes.insert(u.clone(), Sym::atom(hole.clone()));
                let Ok(e) = Self::substitute(pd, &trial) else {
                    continue;
                };
                let Some((c, rest)) = e.linear_in(&hole) else {
                    continue;
                };
                if rest.params().contains(&format!("?{u}")) {
                    continue;
                }
                let difference = ad.sub(&rest);
                let value = match c {
                    1 => Some(difference),
                    -1 => Some(difference.neg()),
                    c => difference.div_exact(c),
                };
                if let Some(value) = value {
                    binding.shapes.insert(u.clone(), value);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if let Some(p) = sig
            .shape_params
            .iter()
            .find(|p| !binding.shapes.contains_key(*p))
        {
            return Err(format!(
                "shape parameter `{p}` is not determined by the arguments; bind it with `{}[{p} = …](…)`",
                sig.name
            ));
        }
        for (pd, ad) in &compound {
            let expected = Self::substitute(pd, &binding)?;
            if !self.prover().zero(&expected.sub(ad)) {
                return Err(format!(
                    "extent `{pd}` is `{expected}` under this binding but the argument has `{ad}`"
                ));
            }
        }
        // Index parameters: symbolic arguments are proved inside the bound; data-dependent
        // arguments keep a runtime obligation.
        for (param, ordinal) in sig.params.iter().zip(&order) {
            let (ValueType::Index { bound }, Some(value)) = (&param.ty, &args[*ordinal].sym) else {
                continue;
            };
            let bound =
                Self::substitute(bound.sym().ok_or("index bound is not symbolic")?, &binding)?;
            let data_dependent = value.params().iter().any(|p| {
                !self.sig.shape_params.contains(p)
                    && self.facts.upper_of(&Atom::Param(p.clone())).is_none()
            });
            let proved = self.prover().nonneg(value) && self.prover().lt(value, &bound);
            if !proved && !data_dependent {
                return Err(format!(
                    "parameter `{}` needs `0 <= {value} < {bound}`, which is not provable here",
                    param.name
                ));
            }
        }
        Ok((order, binding))
    }

    fn user_call(
        &mut self,
        name: &ast::Ident,
        bindings: &[(ast::Ident, ast::Expr)],
        ast_args: &[ast::Arg],
        span: Span,
    ) -> Option<CheckedExpr> {
        let resolved = self.env.resolved;
        let Some(families) = resolved.by_name.get(&name.name) else {
            self.error(name.span, format!("`{}` is not declared", name.name));
            return None;
        };
        let caller_target = self.target.as_deref();
        let mut definitions: Vec<usize> = families
            .iter()
            .flat_map(|f| {
                let family = &resolved.families[*f];
                let has_portable = family.bodies.iter().any(|id| {
                    matches!(
                        resolved.declared[id.0 as usize].kind,
                        DefKind::Body { target: None }
                    )
                });
                let bodies = family.bodies.iter().filter(move |id| {
                    match &resolved.declared[id.0 as usize].kind {
                        DefKind::Body { target: None } => has_portable,
                        DefKind::Body {
                            target: Some(target),
                        } => !has_portable && caller_target == Some(target.as_str()),
                        DefKind::Lower { .. } => false,
                    }
                });
                bodies
                    .chain(family.lowerings.iter().filter(move |_| has_portable))
                    .map(|id| id.0 as usize)
            })
            .collect();
        definitions.sort_unstable();
        if definitions.is_empty() {
            let available: Vec<&str> = families
                .iter()
                .flat_map(|f| resolved.families[*f].bodies.iter())
                .filter_map(|id| resolved.declared[id.0 as usize].kind.target())
                .collect();
            let context = self.target.as_deref().unwrap_or("portable");
            self.error(span, format!("`{}` has no implementation callable from {context} code; backend-specific implementations are available for {}", name.name, available.join(", ")));
            return None;
        }
        if matches!(self.kind, DefKind::Lower { .. })
            && families.contains(&resolved.declared[self.def].family)
        {
            self.error(
                span,
                format!(
                    "a lowering of `{}` cannot recursively call the function it implements",
                    name.name
                ),
            );
            return None;
        }

        // Explicit shape bindings: a shape parameter of this definition passed
        // along as an identity, or a symbolic integer expression.
        let mut explicit: Vec<(String, Sym)> = Vec::new();
        for (param, value) in bindings {
            let extent = match &value.kind {
                A::Name(n)
                    if n.name
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_uppercase())
                        && self.sig.shape_params.contains(&n.name) =>
                {
                    Sym::param(&n.name)
                }
                _ => {
                    let value = self.expr(value, Some(&ValueType::Scalar(DType::I32)))?;
                    let Some(sym) = value.sym.clone() else {
                        self.error(
                            value.span,
                            "a shape binding is a symbolic integer expression",
                        );
                        return None;
                    };
                    sym
                }
            };
            self.numeric_use(&extent);
            explicit.push((param.name.clone(), extent));
        }

        // Arguments are checked once, with scalar hints and write positions from
        // the first definition whose parameter list the call can bind.
        let guide = definitions
            .iter()
            .map(|d| &resolved.declared[*d].sig)
            .find_map(|sig| {
                Self::arg_order(sig, ast_args)
                    .ok()
                    .map(|order| (sig, order))
            });
        let mut args = Vec::new();
        for (ordinal, arg) in ast_args.iter().enumerate() {
            let param = guide.as_ref().and_then(|(sig, order)| {
                order
                    .iter()
                    .position(|o| *o == ordinal)
                    .map(|slot| &sig.params[slot])
            });
            let hint = param
                .and_then(|p| p.ty.scalar_dtype())
                .map(ValueType::Scalar);
            let write_only_candidate =
                param.is_some_and(|p| p.ownership == ParamOwnership::Exclusive);
            args.push(self.expr_inner(&arg.value, hint.as_ref(), write_only_candidate)?);
        }

        let mut candidates: Vec<(usize, Vec<usize>, Binding)> = Vec::new();
        let mut reasons: Vec<String> = Vec::new();
        for d in &definitions {
            match self.bind_candidate(&resolved.declared[*d].sig, &explicit, ast_args, &args) {
                Ok((order, binding)) => candidates.push((*d, order, binding)),
                Err(reason) => reasons.push(reason),
            }
        }
        let Some(family) = candidates
            .first()
            .map(|(d, _, _)| resolved.declared[*d].family)
        else {
            reasons.dedup();
            self.error(
                span,
                format!(
                    "no definition of `{}` accepts these arguments: {}",
                    name.name,
                    reasons.join("; ")
                ),
            );
            return None;
        };
        candidates.retain(|(d, _, _)| resolved.declared[*d].family == family);

        for (argument_ordinal, argument) in args.iter().enumerate() {
            let Some(root) = self.root_var(argument) else {
                continue;
            };
            let LocalKind::Param(own_parameter) = self.kinds[root] else {
                continue;
            };
            if self.sig.params[own_parameter].ownership != ParamOwnership::Exclusive {
                continue;
            }
            let forwarded: Vec<_> = candidates
                .iter()
                .filter_map(|(definition, order, _)| {
                    order
                        .iter()
                        .position(|ordinal| *ordinal == argument_ordinal)
                        .filter(|parameter| {
                            resolved.declared[*definition].sig.params[*parameter].ownership
                                == ParamOwnership::Exclusive
                        })
                        .map(|parameter| (*definition, parameter))
                })
                .collect();
            if forwarded.len() == candidates.len() {
                self.summary.init_passes.push((forwarded, own_parameter));
            }
        }

        // An uninitialized owned tensor may cross an exclusive borrow only when every
        // applicable implementation definitely initializes the whole parameter on all paths.
        for (argument_ordinal, argument) in args.iter().enumerate() {
            let Some(root) = self
                .root_var(argument)
                .filter(|root| self.unassigned.contains(root))
            else {
                continue;
            };
            let full = !self.env.enforce
                || candidates.iter().all(|(definition, order, _)| {
                    order
                        .iter()
                        .position(|ordinal| *ordinal == argument_ordinal)
                        .is_some_and(|parameter| {
                            self.env.summaries[*definition]
                                .full_init
                                .contains(&parameter)
                        })
                });
            if !full {
                self.error(argument.span, format!("`{}` is uninitialized and `{}` does not initialize that exclusive tensor on every path", self.locals[root].name, name.name));
                return None;
            }
            self.unassigned.remove(&root);
        }
        let Some((first, first_order, first_binding)) = candidates.first() else {
            self.error(
                span,
                format!(
                    "no definition of `{}` remains applicable to these arguments",
                    name.name
                ),
            );
            return None;
        };
        let first_sig = &resolved.declared[*first].sig;
        let result = match Self::substitute_ty(&first_sig.result, first_binding) {
            Ok(ty) => ty,
            Err(reason) => {
                self.error(
                    span,
                    format!(
                        "the result of `{}` is not expressible at this call: {reason}",
                        name.name
                    ),
                );
                return None;
            }
        };

        // Logical ownership is checked at the static call boundary. Borrows end
        // with the call in this foundation; `let`-bound view borrows are tracked
        // separately by the body checker.
        let mut accesses: HashMap<LocalId, ParamOwnership> = HashMap::new();
        let mut moved_roots = Vec::new();
        for (parameter, ordinal) in first_sig.params.iter().zip(first_order) {
            let ownership = parameter.ownership;
            if ownership == ParamOwnership::Value {
                continue;
            }
            let argument = &args[*ordinal];
            let Some(root) = self.root_var(argument) else {
                if matches!(ownership, ParamOwnership::Shared | ParamOwnership::Owned)
                    && matches!(argument.ty, ValueType::Tensor(_))
                    && matches!(
                        self.class_of(argument),
                        ValueClass::Computed | ValueClass::Owned
                    )
                {
                    // A fresh logical value (computed, or a rootless owned
                    // materialization/clone/allocation/call result) may be
                    // borrowed or moved into this call. It has no
                    // caller-visible storage root to invalidate or with which
                    // it could alias.
                    continue;
                }
                self.error(
                    argument.span,
                    format!(
                        "parameter `{}` requires a tensor place for {:?} access",
                        parameter.name, ownership
                    ),
                );
                return None;
            };
            if let Some(previous) = accesses.get(&root) {
                let compatible =
                    *previous == ParamOwnership::Shared && ownership == ParamOwnership::Shared;
                if !compatible {
                    self.error(
                        argument.span,
                        format!(
                            "overlapping tensor arguments cannot combine {:?} and {:?} access",
                            previous, ownership
                        ),
                    );
                    return None;
                }
            } else {
                accesses.insert(root, ownership);
            }
            if ownership == ParamOwnership::Owned {
                moved_roots.push(root);
            }
        }
        for root in moved_roots {
            self.moved.insert(root);
        }

        // Calling a backend-specific helper is itself a use of every capability
        // promised by that helper. The caller must therefore declare a superset;
        // portable family calls do not inherit requirements from selectable lowerings.
        let helper_requirements: Vec<_> = candidates
            .iter()
            .filter(|(definition, _, _)| {
                matches!(
                    resolved.declared[*definition].kind,
                    DefKind::Body { target: Some(_) }
                )
            })
            .flat_map(|(definition, _, _)| {
                resolved.declared[*definition]
                    .requires
                    .iter()
                    .map(|(capability, _)| capability.clone())
            })
            .collect();
        for capability in helper_requirements {
            self.use_capability(
                &capability,
                span,
                &format!("backend-specific helper `{}`", name.name),
            );
        }

        // Effects of `inout` parameters.
        let modes: Vec<(crate::sir::Mode, usize)> = first_sig
            .params
            .iter()
            .zip(first_order)
            .map(|(p, o)| (p.mode, *o))
            .collect();
        for (mode, ordinal) in modes {
            let arg = args[ordinal].clone();
            if mode == crate::sir::Mode::In {
                continue;
            }
            let whole = matches!(arg.kind, CheckedExprKind::Local(_));
            let base_local = |operands: &[CheckedExpr]| match operands.first().map(|o| &o.kind) {
                Some(CheckedExprKind::Local(v)) => Some(*v),
                _ => None,
            };
            let binding_local = match &arg.kind {
                CheckedExprKind::Local(v) => Some(*v),
                CheckedExprKind::Primitive {
                    id:
                        PrimitiveId::SliceView { .. } | PrimitiveId::Reshape | PrimitiveId::Transpose,
                    operands,
                } => base_local(operands),
                _ => None,
            };
            let Some(binding_local) = binding_local else {
                self.error(
                    arg.span,
                    "an `inout` argument must name a tensor binding or a selection of one",
                );
                return None;
            };
            let storage_root = self.root_var_local(binding_local);
            self.write(storage_root, binding_local, &[], whole, arg.span)?;
            if whole {
                self.unassigned.remove(&storage_root);
            }
        }

        let mut site = Vec::new();
        for (d, order, binding) in &candidates {
            let sig = &resolved.declared[*d].sig;
            let summary = &self.env.summaries[*d];
            let mut shape_args = Vec::new();
            for p in &sig.shape_params {
                let Some(extent) = binding.shapes.get(p) else {
                    continue;
                };
                if let Some(own) =
                    single_param(extent).filter(|q| self.sig.shape_params.contains(q))
                {
                    self.summary.passes.push((*d, p.clone(), own));
                }
                shape_args.push((p.clone(), extent.clone()));
            }
            let elem_args = sig
                .elem_params
                .iter()
                .filter_map(|p| binding.elems.get(p).map(|e| (p.clone(), e.clone())))
                .collect();
            site.push(CandidateBinding {
                definition: DefId(*d as u32),
                shape_args,
                elem_args,
                arg_order: order.clone(),
                requires_elems: binding.requires.clone(),
            });
            let _ = summary;
        }
        let call = CheckedCall {
            family,
            bindings: site,
            span,
        };
        Some(CheckedExpr::new(
            CheckedExprKind::Call {
                call: Box::new(call),
                args,
            },
            result,
            None,
            span,
        ))
    }
}

fn positional_3(args: &[ast::Arg]) -> bool {
    args.len() == 3 && args.iter().all(|a| a.name.is_none())
}

/// Whether an argument type satisfies one declared capability argument.
/// A rank-two tensor argument matches a rank-two symbolic-axes declaration;
/// scalars match exactly up to the shared float widening of one call.
fn capability_argument_matches(param: &ValueType, arg: &ValueType) -> bool {
    match (param, arg) {
        (ValueType::Scalar(a), ValueType::Scalar(b)) => a == b,
        (ValueType::Tensor(p), ValueType::Tensor(a)) => p.rank() == a.rank(),
        (ValueType::Index { .. }, _) => arg.scalar_dtype() == Some(DType::I32),
        _ => param == arg,
    }
}
