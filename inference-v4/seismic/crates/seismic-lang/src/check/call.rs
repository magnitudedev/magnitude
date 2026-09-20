//! Calls: casts, toolchain operations, target intrinsics, and calls of contract families
//! with one candidate binding per definition that unifies with the arguments.

use super::resolve::{ParamOwnership, Sig};
use super::Checker;
use crate::intrinsics::{self, Intrinsic, IntrinsicParam, IntrinsicResult, Operation, Semantics};
use crate::sir::{
    self, CallId, CallSite, CandidateBinding, DefId, DefKind, Expr, ExprKind, Math, ReduceOp,
    VarId, VarKind,
};
use crate::span::Span;
use crate::sym::{Atom, Sym};
use crate::syntax::ast::{self, BinaryOp, ExprKind as A};
use crate::sir::Mode;
use crate::types::{DType, Elem, Extent, NativeTy, Shaped, Ty};
use std::collections::HashMap;

fn math_of(name: &str) -> Option<(Math, usize)> {
    Some(match name {
        "fma" => (Math::Fma, 3),
        "exp" => (Math::Exp, 1),
        "exp_fast" => (Math::ExpFast, 1),
        "rsqrt" => (Math::Rsqrt, 1),
        "sqrt" => (Math::Sqrt, 1),
        "log" => (Math::Log, 1),
        "sin" => (Math::Sin, 1),
        "cos" => (Math::Cos, 1),
        "abs" => (Math::Abs, 1),
        "max" => (Math::Max, 2),
        "min" => (Math::Min, 2),
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
    shapes: HashMap<String, Extent>,
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
        expected: Option<&Ty>,
        span: Span,
    ) -> Option<Expr> {
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
            "to_owned" | "clone" => {
                if !positional(self, 1) {
                    return None;
                }
                let value = self.expr(&args[0].value, None)?;
                let shape = match (&value.ty, name.name.as_str()) {
                    (Ty::View(shape), "to_owned") => shape.clone(),
                    (Ty::Tile(shape), "to_owned") => shape.clone(),
                    (Ty::Tensor(shape), "clone") => shape.clone(),
                    (found, "to_owned") => {
                        self.error(
                            value.span,
                            format!("`to_owned` materializes a borrowed or computed tensor value, found {found}"),
                        );
                        return None;
                    }
                    (found, _) => {
                        self.error(
                            value.span,
                            format!("`clone` duplicates an owned tensor, found {found}"),
                        );
                        return None;
                    }
                };
                // Temporary bridge: `Load` already denotes a deep snapshot, while
                // the logical result is recorded as owned tensor storage.
                Some(Expr {
                    partial: value.partial,
                    kind: ExprKind::Load(Box::new(value)),
                    ty: Ty::Tensor(shape),
                    sym: None,
                    span,
                })
            }
            "load" => {
                if !positional(self, 1) {
                    return None;
                }
                let v = self.expr(&args[0].value, None)?;
                let (Ty::Tensor(s) | Ty::View(s)) = &v.ty else {
                    self.error(
                        v.span,
                        format!(
                            "`load` snapshots a view in its own representation, found {}",
                            v.ty
                        ),
                    );
                    return None;
                };
                let ty = Ty::Tile(s.clone());
                Some(Expr {
                    partial: v.partial,
                    kind: ExprKind::Load(Box::new(v)),
                    ty,
                    sym: None,
                    span,
                })
            }
            "decode" => {
                if !positional(self, 1) {
                    return None;
                }
                let v = self.expr(&args[0].value, None)?;
                let Some(s) = v.ty.shaped().filter(|s| !matches!(s.elem, Elem::Dtype(_))) else {
                    self.error(v.span, format!("`decode` produces the dense `f32` tile of a packed view, found {}; convert dense values with a cast", v.ty));
                    return None;
                };
                let ty = Ty::Tile(Shaped::new(s.axes.clone(), Elem::Dtype(DType::F32)));
                Some(Expr {
                    partial: v.partial,
                    kind: ExprKind::Decode(Box::new(v)),
                    ty,
                    sym: None,
                    span,
                })
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
                let ty = Ty::Tile(Shaped::new(s.axes, Elem::Dtype(dtype)));
                let value = if name.name == "zeros_like" { 0.0 } else { 1.0 };
                Some(Expr {
                    kind: ExprKind::Filled {
                        like: Box::new(like),
                        value,
                    },
                    ty,
                    sym: None,
                    partial: false,
                    span,
                })
            }
            "select" => {
                if !positional(self, 3) {
                    return None;
                }
                let cond = self.expr(&args[0].value, None)?;
                let then = self.expr(&args[1].value, expected)?;
                let els = self.expr(&args[2].value, Some(&then.ty))?;
                let (shape, dtypes) = self.broadcast(&[&cond, &then, &els], "`select`", span)?;
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
                let Some(dtype) = DType::promote(dtypes[1], dtypes[2]) else {
                    self.error(
                        span,
                        format!(
                            "`select` between {} and {} needs an explicit cast",
                            dtypes[1].name(),
                            dtypes[2].name()
                        ),
                    );
                    return None;
                };
                let ty = self.elementwise(shape, dtype);
                Some(Expr {
                    kind: ExprKind::Select {
                        cond: Box::new(cond),
                        then: Box::new(then),
                        els: Box::new(els),
                    },
                    ty,
                    sym: None,
                    partial: false,
                    span,
                })
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
            "extent" | "capacity" | "valid" => {
                if !positional(self, 2) {
                    return None;
                }
                let geometry = name.name != "extent";
                if geometry && !self.target_form(span, &format!("`{}`", name.name), None) {
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
                let sym = match (&s.axes[axis], geometry) {
                    (Extent::Semantic(sym), false) => {
                        self.numeric_use(sym);
                        sym.clone()
                    }
                    (Extent::Semantic(sym), true) => sym.clone(),
                    (Extent::Structural(slice), false) => {
                        let binder = self.slice_name(*slice);
                        self.error(span, format!("axis {axis} is structural (the width of slice `{binder}`): a structural extent is never a number in portable code, directly or through a generic helper"));
                        return None;
                    }
                    (Extent::Structural(slice), true) => {
                        let capacity = Atom::Param(format!("@capacity#{}", slice.0));
                        self.facts
                            .set_range_lower(capacity.clone(), Sym::constant(1));
                        if name.name == "capacity" {
                            Sym::atom(capacity)
                        } else {
                            let valid = Atom::Param(format!("@valid#{}", slice.0));
                            self.facts.set_range(
                                valid.clone(),
                                Sym::constant(1),
                                Sym::atom(capacity),
                            );
                            Sym::atom(valid)
                        }
                    }
                };
                let kind = if geometry {
                    ExprKind::Geometry {
                        base: Box::new(base),
                        axis,
                        valid: name.name == "valid",
                    }
                } else {
                    ExprKind::ExtentOf {
                        base: Box::new(base),
                        axis,
                    }
                };
                Some(self.scalar(kind, DType::I32, Some(sym), span))
            }
            "coord" => {
                if !positional(self, 1) {
                    return None;
                }
                let var = match &args[0].value.kind {
                    A::Name(n) => self
                        .lookup(&n.name)
                        .filter(|id| self.vars[*id].kind == VarKind::Coordinate),
                    _ => None,
                };
                let Some(var) = var else {
                    self.error(
                        span,
                        "`coord(i)` takes a tile coordinate bound by `owned` or `axis`",
                    );
                    return None;
                };
                // Over an axis that a caller may bind structurally the coordinate is a member of
                // the caller's domain, not of `0..P`: it carries no bounds facts.
                let generic = matches!(&self.vars[var].ty, Ty::Index(bound) if single_param(bound).is_some_and(|p| self.sig.shape_params.contains(&p)));
                let sym = if generic {
                    None
                } else {
                    self.atoms.get(&var).map(|a| Sym::atom(a.clone()))
                };
                Some(self.scalar(ExprKind::CoordOf(var), DType::I32, sym, span))
            }
            "atomic" => {
                if !positional(self, 3) || !self.target_form(span, "`atomic`", None) {
                    return None;
                }
                let op = match &args[0].value.kind {
                    A::Name(n) if n.name == "add" => BinaryOp::Add,
                    A::Name(n) if matches!(n.name.as_str(), "max" | "min") => {
                        self.error(n.span, format!("`atomic({}, …)` has no structured IR form; only `atomic(add, place, value)` is representable", n.name));
                        return None;
                    }
                    _ => {
                        self.error(args[0].value.span, "`atomic` needs an operation name: add");
                        return None;
                    }
                };
                let place = self.place(&args[1].value)?;
                let Ty::Scalar(dtype) = place.ty else {
                    self.error(
                        place.span,
                        format!("the `atomic` place selects one element, found {}", place.ty),
                    );
                    return None;
                };
                let value = self.expr(&args[2].value, Some(&Ty::Scalar(dtype)))?;
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
                self.forbid_partial(&value, "an atomic update");
                let root = self.root_var(&place)?;
                if !matches!(self.vars[root].kind, VarKind::Param(i) if self.sig.params[i].mode != Mode::In)
                    && self.vars[root].kind != VarKind::State
                {
                    self.error(
                        place.span,
                        "an `atomic` place is an `out`/`inout` parameter or local state",
                    );
                    return None;
                }
                self.published.insert(root);
                self.mutated.push(root);
                Some(Expr {
                    kind: ExprKind::Atomic {
                        op,
                        place: Box::new(place),
                        value: Box::new(value),
                    },
                    ty: Ty::Void,
                    sym: None,
                    partial: false,
                    span,
                })
            }
            "owned" | "axis" | "lanes" => {
                self.error(
                    span,
                    format!("`{}` is an iterator and appears only in `for`", name.name),
                );
                None
            }
            "full" => {
                self.error(span, "`full(X)` is a `where` predicate");
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
        op: Math,
        arity: usize,
        name: &str,
        args: &[ast::Arg],
        expected: Option<&Ty>,
        span: Span,
    ) -> Option<Expr> {
        if args.len() != arity || args.iter().any(|a| a.name.is_some()) {
            self.error(
                span,
                format!("`{name}` takes {arity} positional argument(s)"),
            );
            return None;
        }
        let float_only = !matches!(op, Math::Max | Math::Min | Math::Abs);
        let default = Ty::Scalar(DType::F32);
        let mut hint: Option<Ty> = expected
            .cloned()
            .or_else(|| float_only.then(|| default.clone()));
        let mut out: Vec<Expr> = Vec::new();
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
        let operands: Vec<&Expr> = out.iter().collect();
        let (shape, dtypes) = self.broadcast(&operands, &format!("`{name}`"), span)?;
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
        let ty = self.elementwise(shape, dtype);
        Some(Expr {
            kind: ExprKind::Math { op, args: out },
            ty,
            sym: None,
            partial: false,
            span,
        })
    }

    /// Build `max`/`min` from checked operands (state accumulation).
    pub fn math_exprs(&mut self, op: Math, l: Expr, r: Expr, span: Span) -> Option<Expr> {
        let (shape, dtypes) = self.broadcast(&[&l, &r], "`max`/`min`", span)?;
        let Some(dtype) = DType::promote(dtypes[0], dtypes[1]).filter(|d| d.is_numeric()) else {
            self.error(
                span,
                format!(
                    "`max`/`min` between {} and {} needs an explicit cast",
                    dtypes[0].name(),
                    dtypes[1].name()
                ),
            );
            return None;
        };
        let ty = self.elementwise(shape, dtype);
        Some(Expr {
            kind: ExprKind::Math {
                op,
                args: vec![l, r],
            },
            ty,
            sym: None,
            partial: false,
            span,
        })
    }

    fn reduce(&mut self, args: &[ast::Arg], unordered: bool, span: Span) -> Option<Expr> {
        let t = self.expr(&args[0].value, None)?;
        let Ty::Tile(s) = &t.ty else {
            self.error(
                t.span,
                format!(
                    "`reduce` reduces a dense tile, found {}; read views with `load` or a cast",
                    t.ty
                ),
            );
            return None;
        };
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
                "sum" => Some(ReduceOp::Sum),
                "max" => Some(ReduceOp::Max),
                "min" => Some(ReduceOp::Min),
                "argmax" => Some(ReduceOp::Argmax),
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
        if unordered && op == ReduceOp::Argmax {
            self.error(span, "`argmax` has one defined winner (ties go to the smaller index) and never accepts `unordered`");
            return None;
        }
        self.forbid_partial(&t, "a reduction operand");
        // Over a structural axis the value is one aggregate per tuned piece.
        let partial = match &s.axes[axis] {
            Extent::Structural(_) => true,
            Extent::Semantic(extent) => {
                if let Some(p) = single_param(extent).filter(|p| self.sig.shape_params.contains(p))
                {
                    self.summary.reduces.insert(p);
                }
                false
            }
        };
        let elem = if op == ReduceOp::Argmax {
            DType::I32
        } else {
            dtype
        };
        let mut axes = s.axes.clone();
        axes.remove(axis);
        let ty = if axes.is_empty() {
            Ty::Scalar(elem)
        } else {
            Ty::Tile(Shaped::new(axes, Elem::Dtype(elem)))
        };
        Some(Expr {
            kind: ExprKind::Reduce {
                value: Box::new(t),
                axis,
                op,
                unordered,
            },
            ty,
            sym: None,
            partial,
            span,
        })
    }

    fn reshape(&mut self, args: &[ast::Arg], span: Span) -> Option<Expr> {
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
            let Extent::Semantic(extent) = axis else {
                self.error(span, "`reshape` cannot expose a product of structural extents or erase a partition identity; every source axis must be semantic");
                return None;
            };
            source = source.mul(extent);
        }
        let mut axes = Vec::new();
        let mut target = Sym::constant(1);
        for dimension in dimensions {
            let d = self.expr(dimension, Some(&Ty::Scalar(DType::I32)))?;
            let Some(extent) = d.sym.clone() else {
                self.error(d.span, "`reshape` extents are symbolic integer expressions");
                return None;
            };
            self.require_nonneg(&extent, d.span, "reshape extent may be negative");
            target = target.mul(&extent);
            axes.push(Extent::Semantic(extent));
        }
        if axes.is_empty() || !self.prover().zero(&target.sub(&source)) {
            self.error(span, format!("`reshape` must provably preserve element correspondence: `{source}` elements into `{target}`"));
            return None;
        }
        let shaped = Shaped::new(axes.clone(), s.elem);
        let ty = if matches!(base.ty, Ty::Tensor(_)) {
            Ty::Tensor(shaped)
        } else {
            Ty::View(shaped)
        };
        Some(Expr {
            partial: base.partial,
            kind: ExprKind::Reshape {
                base: Box::new(base),
                axes,
            },
            ty,
            sym: None,
            span,
        })
    }

    // ---- target intrinsics ----

    fn intrinsic(
        &mut self,
        backend: &ast::Ident,
        capability: &ast::Ident,
        name: &ast::Ident,
        args: &[ast::Arg],
        span: Span,
    ) -> Option<Expr> {
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
        let Some(intrinsic) = intrinsics::lookup(&backend.name, &capability.name, &name.name)
        else {
            self.error(
                name.span,
                format!(
                    "capability `{}` has no intrinsic `{}`",
                    capability_id.path(),
                    name.name
                ),
            );
            return None;
        };
        self.use_capability(
            &capability_id,
            span,
            &format!("intrinsic `{}`", intrinsic.id.path()),
        );
        self.intrinsic_call(&intrinsic, args, span)
    }

    fn intrinsic_call(
        &mut self,
        intrinsic: &Intrinsic,
        args: &[ast::Arg],
        span: Span,
    ) -> Option<Expr> {
        let target = intrinsic.id.capability.backend.as_str();
        let operation = intrinsic.operation;
        if matches!(
            operation,
            Operation::MatrixMatmul | Operation::MatrixMatmulAdd
        ) {
            return self.matrix_intrinsic(intrinsic, args, span);
        }
        if args.len() != intrinsic.params.len() || args.iter().any(|a| a.name.is_some()) {
            self.error(
                span,
                format!(
                    "`{}` takes {} positional argument(s)",
                    intrinsic.id.path(),
                    intrinsic.params.len()
                ),
            );
            return None;
        }
        let fragment = |ty: &Ty| matches!(ty, Ty::Native(NativeTy { target: t, name, .. }) if t == target && name == Operation::Matrix.name());
        let mut out = Vec::new();
        let mut float_dtype: Option<DType> = None;
        let mut named_dtype: Option<DType> = None;
        for (arg, param) in args.iter().zip(&intrinsic.params) {
            match param {
                IntrinsicParam::DTypeName => {
                    let dtype = match &arg.value.kind {
                        A::Name(n) => DType::from_name(&n.name),
                        _ => None,
                    };
                    let Some(d) = dtype else {
                        self.error(arg.value.span, "expected a dtype name");
                        return None;
                    };
                    named_dtype = Some(d);
                    out.push(self.scalar(ExprKind::Int(0), d, None, arg.value.span));
                }
                IntrinsicParam::FloatScalar => {
                    let e = self.expr(&arg.value, float_dtype.map(Ty::Scalar).as_ref())?;
                    let Some(d) = e.ty.scalar_dtype().filter(|d| d.is_float()) else {
                        self.error(
                            e.span,
                            format!(
                                "`{target}.{operation}` needs a float scalar, found {}",
                                e.ty
                            ),
                        );
                        return None;
                    };
                    float_dtype = Some(d);
                    out.push(e);
                }
                IntrinsicParam::Frag8x8 => {
                    let e = self.expr(&arg.value, None)?;
                    if !fragment(&e.ty) {
                        self.error(e.span, format!("`{target}.{operation}` needs a `{target}.{}` value, found {}; native types of different targets are not interchangeable", Operation::Matrix.name(), e.ty));
                        return None;
                    }
                    out.push(e);
                }
                IntrinsicParam::Tile2 => {
                    // A block store overwrites its window; it does not read the tile.
                    let e =
                        self.expr_inner(&arg.value, None, operation == Operation::MatrixStore)?;
                    if !matches!(&e.ty, Ty::Tile(s) | Ty::View(s) if s.rank() == 2) {
                        self.error(
                            e.span,
                            format!(
                                "`{target}.{operation}` needs a rank-2 tile or view, found {}",
                                e.ty
                            ),
                        );
                        return None;
                    }
                    out.push(e);
                }
                IntrinsicParam::Integer => {
                    let e = self.expr(&arg.value, Some(&Ty::Scalar(DType::I32)))?;
                    if !e.ty.scalar_dtype().is_some_and(DType::is_int) {
                        self.error(e.span, "a participant index is a 32-bit integer");
                        return None;
                    }
                    out.push(e);
                }
                IntrinsicParam::Int => {
                    let e = self.expr(&arg.value, Some(&Ty::Scalar(DType::I32)))?;
                    if e.sym.is_none() {
                        self.error(
                            e.span,
                            format!("`{target}.{operation}` needs a symbolic integer offset"),
                        );
                        return None;
                    }
                    out.push(e);
                }
                IntrinsicParam::MatrixOperand => {
                    unreachable!("logical matrix intrinsics are checked separately")
                }
            }
        }
        for e in &out {
            self.forbid_partial(e, "an intrinsic operand");
        }
        // Block loads and stores touch a fixed window at (row, column): prove it fits the
        // semantic axes; structural axes are the target's own geometry.
        if let Semantics::Load { rows, columns, .. } | Semantics::Store { rows, columns } =
            operation.semantics()
        {
            let axes = out[1]
                .ty
                .shaped()
                .map(|s| s.axes.clone())
                .unwrap_or_default();
            for (k, (axis, window)) in axes.iter().zip([rows, columns]).enumerate() {
                let (Extent::Semantic(dim), Some(offset)) = (axis, out[2 + k].sym.clone()) else {
                    continue;
                };
                let interval = self.prover().interval(&offset);
                let offset_span = out[2 + k].span;
                self.require_nonneg(&interval.lo, offset_span, "block offset may be negative");
                self.require_nonneg(
                    &dim.sub(&interval.hi).sub(&Sym::constant(window as i64)),
                    offset_span,
                    &format!("{window}-wide block may exceed extent `{dim}`"),
                );
            }
        }
        for written in operation.writes_arguments() {
            let place = out[*written].clone();
            let root = self.write(&place, place.span, false)?;
            // Definite assignment tracks whole tiles: only a store covering the tile initializes it.
            if operation == Operation::MatrixStore {
                let covers = place.ty.shaped().is_some_and(|s| {
                    s.axes
                        .iter()
                        .all(|a| a.semantic().and_then(Sym::as_constant) == Some(8))
                }) && out[2..4]
                    .iter()
                    .all(|e| e.sym.as_ref().and_then(Sym::as_constant) == Some(0));
                if covers && matches!(place.kind, ExprKind::Var(_)) {
                    self.unassigned.remove(&root);
                }
            }
        }
        let ty = match intrinsic.result {
            IntrinsicResult::Void => Ty::Void,
            IntrinsicResult::Integer => Ty::Scalar(DType::I32),
            IntrinsicResult::FloatScalar => Ty::Scalar(float_dtype.unwrap_or(DType::F32)),
            IntrinsicResult::Frag8x8OfNamedDtype => {
                let Semantics::Fragment { rows, columns } = operation.semantics() else {
                    self.error(
                        span,
                        format!("`{target}.{operation}` does not declare a native fragment"),
                    );
                    return None;
                };
                out.clear();
                Ty::Native(NativeTy {
                    target: target.to_string(),
                    name: operation.name().to_string(),
                    shape: vec![Sym::constant(rows as i64), Sym::constant(columns as i64)],
                    elem: Some(Elem::Dtype(named_dtype.unwrap_or(DType::F32))),
                })
            }
            IntrinsicResult::LogicalMatrix => {
                unreachable!("logical matrix intrinsics are checked separately")
            }
        };
        let expression = Expr {
            kind: ExprKind::Intrinsic {
                op: operation,
                args: out,
            },
            ty,
            sym: None,
            partial: false,
            span,
        };
        self.record_intrinsic(intrinsic, &expression);
        Some(expression)
    }

    fn record_intrinsic(&mut self, intrinsic: &Intrinsic, expression: &Expr) {
        let ExprKind::Intrinsic { args, .. } = &expression.kind else {
            return;
        };
        let used = sir::IntrinsicUse {
            id: intrinsic.id.clone(),
            operation: intrinsic.operation,
            arguments: args.iter().map(|argument| argument.ty.clone()).collect(),
            result: expression.ty.clone(),
        };
        if !self.intrinsic_uses.contains(&used) {
            self.intrinsic_uses.push(used);
        }
    }

    fn matrix_intrinsic(
        &mut self,
        intrinsic: &Intrinsic,
        args: &[ast::Arg],
        span: Span,
    ) -> Option<Expr> {
        let operation = intrinsic.operation;
        let mut positional = Vec::new();
        let mut accumulation = None;
        for argument in args {
            match argument.name.as_ref().map(|name| name.name.as_str()) {
                None => positional.push(&argument.value),
                Some("accumulation") if operation == Operation::MatrixMatmul => {
                    if accumulation.replace(&argument.value).is_some() {
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
                        format!("`{}` has no named argument `{name}`", intrinsic.id.path()),
                    );
                    return None;
                }
            }
        }
        let expected = if operation == Operation::MatrixMatmul {
            2
        } else {
            3
        };
        if positional.len() != expected {
            self.error(
                span,
                format!(
                    "`{}` takes {expected} positional argument(s)",
                    intrinsic.id.path()
                ),
            );
            return None;
        }
        let mut checked = Vec::new();
        for operand in positional {
            let expression = self.expr(operand, None)?;
            if !matches!(&expression.ty, Ty::Tensor(shape) | Ty::View(shape) | Ty::Tile(shape) if shape.rank() == 2)
            {
                self.error(
                    expression.span,
                    format!(
                        "`{}` needs rank-two logical tensor operands, found {}",
                        intrinsic.id.path(),
                        expression.ty
                    ),
                );
                return None;
            }
            self.forbid_partial(&expression, "a logical matrix intrinsic operand");
            checked.push(expression);
        }
        let left = checked[0].ty.shaped().expect("checked rank-two operand");
        let right = checked[1].ty.shaped().expect("checked rank-two operand");
        if !self.same_extent(&left.axes[1], &right.axes[0]) {
            self.error(
                span,
                format!(
                    "`{}` inner axes differ: {} versus {}",
                    intrinsic.id.path(),
                    left.axes[1],
                    right.axes[0]
                ),
            );
            return None;
        }
        let (element, result_axes) = if operation == Operation::MatrixMatmul {
            let Some(accumulation) = accumulation else {
                self.error(
                    span,
                    "`matrix.matmul` requires the named argument `accumulation=<dtype>`",
                );
                return None;
            };
            let dtype = match &accumulation.kind {
                A::Name(name) => DType::from_name(&name.name),
                _ => None,
            };
            let Some(dtype) = dtype.filter(|dtype| dtype.is_numeric()) else {
                self.error(
                    accumulation.span,
                    "matrix accumulation must name a numeric dtype",
                );
                return None;
            };
            (
                Elem::Dtype(dtype),
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
                        intrinsic.id.path()
                    ),
                );
                return None;
            }
            (accumulator.elem.clone(), accumulator.axes.clone())
        };
        let expression = Expr {
            kind: ExprKind::Intrinsic {
                op: operation,
                args: checked,
            },
            ty: Ty::Tensor(Shaped::new(result_axes, element)),
            sym: None,
            partial: false,
            span,
        };
        self.record_intrinsic(intrinsic, &expression);
        Some(expression)
    }

    // ---- contract families ----

    /// Parameter ordinal -> argument ordinal. `into=` binds the one remaining `out`/`inout`
    /// parameter when no parameter is named `into`.
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
                .filter(|i| order[*i].is_none() && sig.params[*i].mode != Mode::In)
                .collect();
            let [slot] = open.as_slice() else {
                return Err("`into=` binds the one unbound `out`/`inout` parameter".to_string());
            };
            order[*slot] = Some(ordinal);
        }
        order
            .into_iter()
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| "leaves a parameter unbound".to_string())
    }

    /// Substitute a candidate's bound shape parameters into one of its symbolic extents.
    fn substitute(s: &Sym, binding: &Binding) -> Result<Sym, String> {
        for p in s.params() {
            match binding.shapes.get(&p) {
                Some(Extent::Semantic(_)) => {}
                Some(Extent::Structural(_)) => {
                    return Err(format!(
                        "`{s}` computes with `{p}`, which is bound to a structural extent"
                    ))
                }
                None => {
                    return Err(format!(
                        "shape parameter `{p}` is not determined by the arguments"
                    ))
                }
            }
        }
        Ok(substitute_all(s, &|p| match binding.shapes.get(p) {
            Some(Extent::Semantic(v)) => Some(v.clone()),
            _ => None,
        }))
    }

    fn substitute_ty(ty: &Ty, binding: &Binding) -> Result<Ty, String> {
        let shaped = |s: &Shaped| -> Result<Shaped, String> {
            let mut axes = Vec::new();
            for axis in &s.axes {
                axes.push(match axis {
                    Extent::Semantic(sym) => {
                        match single_param(sym).and_then(|p| binding.shapes.get(&p)) {
                            Some(extent) => extent.clone(),
                            None => Extent::Semantic(Self::substitute(sym, binding)?),
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
            Ok(Shaped::new(axes, elem))
        };
        Ok(match ty {
            Ty::Tensor(s) => Ty::Tensor(shaped(s)?),
            Ty::View(s) => Ty::View(shaped(s)?),
            Ty::Tile(s) => Ty::Tile(shaped(s)?),
            Ty::Index(n) => Ty::Index(Self::substitute(n, binding)?),
            Ty::Range(n) => Ty::Range(Self::substitute(n, binding)?),
            Ty::Tuple(items) => Ty::Tuple(
                items
                    .iter()
                    .map(|t| Self::substitute_ty(t, binding))
                    .collect::<Result<_, _>>()?,
            ),
            other => other.clone(),
        })
    }

    fn unify(
        &self,
        param: &Ty,
        arg: &Ty,
        binding: &mut Binding,
        compound: &mut Vec<(Sym, Extent)>,
    ) -> Result<(), String> {
        let mismatch = || Err(format!("expects {param} but was given {arg}"));
        match (param, arg) {
            (Ty::Scalar(a), _) => match arg.scalar_dtype() {
                Some(b) if *a == b || (a.is_float() && b.is_float()) => Ok(()),
                _ => mismatch(),
            },
            (Ty::Index(_), _) if arg.scalar_dtype() == Some(DType::I32) => Ok(()),
            (Ty::Range(p), Ty::Range(a)) => match single_param(p) {
                Some(name) => match binding.shapes.get(&name) {
                    Some(bound) if !self.prover().zero(&bound.semantic().unwrap().sub(a)) => {
                        mismatch()
                    }
                    Some(_) => Ok(()),
                    None => {
                        binding.shapes.insert(name, Extent::Semantic(a.clone()));
                        Ok(())
                    }
                },
                None if self.prover().zero(&p.sub(a)) => Ok(()),
                None => mismatch(),
            },
            (Ty::Tensor(p), Ty::Tensor(a) | Ty::View(a))
            | (Ty::View(p), Ty::Tensor(a) | Ty::View(a) | Ty::Tile(a))
            | (Ty::Tile(p), Ty::Tile(a)) => {
                if p.rank() != a.rank() {
                    return mismatch();
                }
                for (pd, ad) in p.axes.iter().zip(&a.axes) {
                    let Extent::Semantic(pd) = pd else {
                        return mismatch();
                    };
                    match single_param(pd) {
                        Some(name) => match binding.shapes.get(&name) {
                            Some(bound) if !self.same_extent(bound, ad) => return Err(format!("binds shape parameter `{name}` to both {bound} and {ad}; equal widths of unrelated slices establish nothing")),
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
            (Ty::Tuple(p), Ty::Tuple(a)) if p.len() == a.len() => p
                .iter()
                .zip(a)
                .try_for_each(|(p, a)| self.unify(p, a, binding, compound)),
            (Ty::Native(p), Ty::Native(a)) if p == a => Ok(()),
            _ => mismatch(),
        }
    }

    /// Bind one candidate definition to the checked arguments.
    fn bind_candidate(
        &self,
        sig: &Sig,
        explicit: &[(String, Extent)],
        ast_args: &[ast::Arg],
        args: &[Expr],
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
            // An `out`/`inout` tile parameter binds a view of local tile state as that sub-tile:
            // the callee updates it in place (disjointness of parallel writers is checked at
            // the write, like any other non-whole place).
            let argument = &args[*ordinal];
            let sub_tile = match (&param.ty, &argument.ty) {
                (Ty::Tile(_), Ty::View(shaped))
                    if param.mode != Mode::In
                        && self.root_var(argument).is_some_and(|root| {
                            matches!(self.vars[root].kind, VarKind::State)
                                && matches!(self.vars[root].ty, Ty::Tile(_))
                        }) =>
                {
                    Some(Ty::Tile(shaped.clone()))
                }
                _ => None,
            };
            self.unify(
                &param.ty,
                sub_tile.as_ref().unwrap_or(&argument.ty),
                &mut binding,
                &mut compound,
            )
            .map_err(|e| format!("parameter `{}` {e}", param.name))?;
        }
        // Compound extents with one unknown, linear in it, determine that parameter.
        loop {
            let mut changed = false;
            for (pd, ad) in &compound {
                let Extent::Semantic(ad) = ad else { continue };
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
                trial
                    .shapes
                    .insert(u.clone(), Extent::Semantic(Sym::atom(hole.clone())));
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
                    binding.shapes.insert(u.clone(), Extent::Semantic(value));
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
            return Err(format!("shape parameter `{p}` is not determined by the arguments; bind it with `{}[{p} = …](…)`", sig.name));
        }
        for (pd, ad) in &compound {
            let expected = Self::substitute(pd, &binding)?;
            if !self.same_extent(&Extent::Semantic(expected.clone()), ad) {
                return Err(format!(
                    "extent `{pd}` is `{expected}` under this binding but the argument has `{ad}`"
                ));
            }
        }
        // Index parameters: symbolic arguments are proved inside the bound; data-dependent
        // arguments keep a runtime obligation.
        for (param, ordinal) in sig.params.iter().zip(&order) {
            let (Ty::Index(bound), Some(value)) = (&param.ty, &args[*ordinal].sym) else {
                continue;
            };
            let bound = Self::substitute(bound, &binding)?;
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
    ) -> Option<Expr> {
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

        // Explicit shape bindings: a slice or an own shape parameter is passed along as an
        // identity; anything else is a symbolic integer.
        let mut explicit: Vec<(String, Extent)> = Vec::new();
        for (param, value) in bindings {
            let identity = match &value.kind {
                A::Name(n) => match self.lookup(&n.name).map(|id| self.vars[id].ty.clone()) {
                    Some(Ty::Slice(slice)) => Some(Extent::Structural(slice)),
                    None if self.sig.shape_params.contains(&n.name) => {
                        Some(Extent::Semantic(Sym::param(&n.name)))
                    }
                    _ => None,
                },
                _ => None,
            };
            let extent = match identity {
                Some(extent) => extent,
                None => {
                    let value = self.expr(value, Some(&Ty::Scalar(DType::I32)))?;
                    let Some(sym) = value.sym.clone() else {
                        self.error(
                            value.span,
                            "a shape binding is a slice or a symbolic integer expression",
                        );
                        return None;
                    };
                    Extent::Semantic(sym)
                }
            };
            explicit.push((param.name.clone(), extent));
        }

        // Arguments are checked once, with scalar hints and `out` positions from the first
        // definition whose parameter list the call can bind.
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
            let hint = param.and_then(|p| p.ty.scalar_dtype()).map(Ty::Scalar);
            let write_only_candidate = param.is_some_and(|p| {
                p.mode == Mode::Out || p.ownership == ParamOwnership::Exclusive
            });
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
            let Some(root) = self.root_var(argument) else { continue };
            let VarKind::Param(own_parameter) = self.vars[root].kind else { continue };
            if self.sig.params[own_parameter].ownership != ParamOwnership::Exclusive { continue; }
            let forwarded: Vec<_> = candidates.iter().filter_map(|(definition, order, _)| {
                order.iter().position(|ordinal| *ordinal == argument_ordinal)
                    .filter(|parameter| resolved.declared[*definition].sig.params[*parameter].ownership == ParamOwnership::Exclusive)
                    .map(|parameter| (*definition, parameter))
            }).collect();
            if forwarded.len() == candidates.len() {
                self.summary.init_passes.push((forwarded, own_parameter));
            }
        }

        // An uninitialized owned tensor may cross an exclusive borrow only when every
        // applicable implementation definitely initializes the whole parameter on all paths.
        for (argument_ordinal, argument) in args.iter().enumerate() {
            let Some(root) = self.root_var(argument).filter(|root| self.unassigned.contains(root)) else { continue };
            let full = !self.env.enforce || candidates.iter().all(|(definition, order, _)| {
                order.iter().position(|ordinal| *ordinal == argument_ordinal)
                    .is_some_and(|parameter| self.env.summaries[*definition].full_init.contains(&parameter))
            });
            if !full {
                self.error(argument.span, format!("`{}` is uninitialized and `{}` does not initialize that exclusive tensor on every path", self.vars[root].name, name.name));
                return None;
            }
            self.unassigned.remove(&root);
        }

        // A structural extent is never a number, also through a generic helper.
        if self.env.enforce {
            let mut rejected: Vec<String> = Vec::new();
            candidates.retain(|(d, _, binding)| {
                let numeric = &self.env.summaries[*d].numeric;
                let offending: Vec<&String> = binding
                    .shapes
                    .iter()
                    .filter(|(p, extent)| {
                        matches!(extent, Extent::Structural(_)) && numeric.contains(*p)
                    })
                    .map(|(p, _)| p)
                    .collect();
                for p in &offending {
                    rejected.push(format!("`{p}`"));
                }
                offending.is_empty()
            });
            let implemented = !candidates.is_empty();
            if !rejected.is_empty() && !implemented {
                rejected.sort();
                rejected.dedup();
                self.error(span, format!("`{}` uses shape parameter {} as a number (arithmetic, `extent`, a range bound or a coordinate), but this call binds it to a structural extent: the width of a slice is never a number, directly or through a generic helper", name.name, rejected.join(", ")));
                return None;
            }
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
        // with the call in this foundation; `let`-bound slice borrows are tracked
        // separately by the body checker.
        let mut accesses: HashMap<VarId, ParamOwnership> = HashMap::new();
        let mut moved_roots = Vec::new();
        for (parameter, ordinal) in first_sig.params.iter().zip(first_order) {
            let ownership = parameter.ownership;
            if ownership == ParamOwnership::Value {
                continue;
            }
            let argument = &args[*ordinal];
            let Some(root) = self.root_var(argument) else {
                if matches!(ownership, ParamOwnership::Shared | ParamOwnership::Owned)
                    && matches!(argument.ty, Ty::Tensor(_) | Ty::View(_))
                {
                    // A fresh logical value may be borrowed or moved into this
                    // call. It has no caller-visible storage root to invalidate or
                    // with which it could alias.
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

        // Effects of `out`/`inout` parameters.
        let modes: Vec<(Mode, usize)> = first_sig
            .params
            .iter()
            .zip(first_order)
            .map(|(p, o)| (p.mode, *o))
            .collect();
        for (mode, ordinal) in modes {
            let arg = args[ordinal].clone();
            self.forbid_partial(&arg, &format!("an argument of `{}`", name.name));
            if mode == Mode::In {
                continue;
            }
            let whole = matches!(arg.kind, ExprKind::Var(_));
            let root = self.write(&arg, arg.span, whole)?;
            if whole {
                self.unassigned.remove(&root);
            }
        }

        let mut partial = false;
        let mut site = Vec::new();
        for (d, order, binding) in &candidates {
            let sig = &resolved.declared[*d].sig;
            let summary = &self.env.summaries[*d];
            let mut shape_args = Vec::new();
            for p in &sig.shape_params {
                let Some(extent) = binding.shapes.get(p) else {
                    continue;
                };
                match extent {
                    Extent::Structural(_) => partial |= summary.reduces.contains(p),
                    Extent::Semantic(sym) => {
                        if let Some(own) =
                            single_param(sym).filter(|q| self.sig.shape_params.contains(q))
                        {
                            self.summary.passes.push((*d, p.clone(), own));
                        }
                    }
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
        }
        let call = CallId(self.calls.len() as u32);
        self.calls.push(CallSite {
            family,
            bindings: site,
            result: result.clone(),
            span,
        });
        Some(sir::Expr {
            kind: ExprKind::Call { call, args },
            ty: result,
            sym: None,
            partial,
            span,
        })
    }
}
