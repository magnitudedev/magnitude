//! Calls: casts, toolchain operations, capability intrinsics, and calls of
//! contract families. A family call is checked once against its contract:
//! the family's dimension plan solves the callee dimensions from the actual
//! axes (L16), and the members that apply at the call are recorded.

use super::dimensions::{self, DimensionCallError};
use super::ir::{
    Call as CheckedCall, CallContext, Candidate, Expr as CheckedExpr,
    ExprKind as CheckedExprKind, Index as CheckedIndex, IntrinsicOverload, LocalId,
    Ownership as ParamOwnership, Predicate,
};
use super::resolve::SigParam;
use super::{Checker, ValueClass};
use crate::checked::{DiagnosticRule, ElementTarget};
use crate::expr::{AnyExpr, ExprArena, IntExpr, SymbolId};
use crate::intrinsics::{self, primitive, reduction_result, MathOp, PrimitiveId, RepresentationTarget};
use crate::registry::{
    IntrinsicExecution, IntrinsicParticipation, IntrinsicResultType, IntrinsicSignature,
    OperandCategory, RepresentationKind,
};
use crate::span::Span;
use crate::syntax::ast::{self, ExprKind as A};
use crate::types::{DType, Elem, TensorType, ValueType};
use std::collections::HashMap;

/// Element bindings of one family member at a call.
#[derive(Default)]
struct Elements {
    /// Member element parameter -> element (possibly a caller parameter).
    arguments: HashMap<String, Elem>,
    /// Caller element parameters this member requires to be a concrete element.
    requires: Vec<(String, Elem)>,
}

impl Elements {
    /// Unify one formal element with one actual element.
    fn unify(&mut self, formal: &Elem, actual: &Elem) -> bool {
        match (formal, actual) {
            (Elem::Param(name), actual) => match self.arguments.get(name) {
                Some(bound) => bound == actual,
                None => {
                    let admitted = !matches!(actual, Elem::Dtype(d) if !d.is_float());
                    if admitted {
                        self.arguments.insert(name.clone(), actual.clone());
                    }
                    admitted
                }
            },
            // A concrete member element against a caller element parameter
            // applies where that parameter is this element.
            (concrete, Elem::Param(own)) => match self.requires.iter().find(|(p, _)| p == own) {
                Some((_, required)) => required == concrete,
                None => {
                    self.requires.push((own.clone(), concrete.clone()));
                    true
                }
            },
            (x, y) => x == y,
        }
    }

    /// Unify every tensor element of a formal type with the actual type.
    fn unify_type(&mut self, formal: &ValueType, actual: &ValueType) -> bool {
        match (formal, actual) {
            (ValueType::Tensor(f), ValueType::Tensor(a)) => self.unify(&f.elem, &a.elem),
            (ValueType::Tuple(f), ValueType::Tuple(a)) if f.len() == a.len() => {
                f.iter().zip(a.iter()).all(|(f, a)| self.unify_type(f, a))
            }
            _ => true,
        }
    }
}

/// The member's formal types mention only its dimensions; map them to the
/// call's dimension values in the caller's arena.
fn substitute(
    member: &ExprArena,
    dimensions: &[SymbolId],
    values: &[IntExpr],
    caller: &mut ExprArena,
    expression: IntExpr,
) -> IntExpr {
    let mut map = |symbol: SymbolId, _: &mut ExprArena| {
        let ordinal = dimensions
            .iter()
            .position(|candidate| *candidate == symbol)
            .unwrap_or_else(|| panic!("a signature expression mentions a non-dimension symbol"));
        AnyExpr::Int(values[ordinal])
    };
    super::xfer::transfer_int(member, expression, caller, &mut map)
}

fn substitute_type(
    member: &ExprArena,
    dimensions: &[SymbolId],
    values: &[IntExpr],
    elements: &HashMap<String, Elem>,
    caller: &mut ExprArena,
    ty: &ValueType,
) -> ValueType {
    match ty {
        ValueType::Tensor(tensor) => ValueType::Tensor(TensorType::new(
            tensor
                .axes
                .iter()
                .map(|axis| substitute(member, dimensions, values, caller, *axis))
                .collect(),
            match &tensor.elem {
                Elem::Param(name) => elements
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| Elem::Param(name.clone())),
                other => other.clone(),
            },
        )),
        ValueType::Index { bound } => ValueType::Index {
            bound: substitute(member, dimensions, values, caller, *bound),
        },
        ValueType::Range { bound } => ValueType::Range {
            bound: substitute(member, dimensions, values, caller, *bound),
        },
        ValueType::Tuple(items) => ValueType::Tuple(
            crate::types::NonEmpty::new(
                items
                    .iter()
                    .map(|item| substitute_type(member, dimensions, values, elements, caller, item))
                    .collect(),
            )
            .expect("a checked tuple is nonempty"),
        ),
        other => other.clone(),
    }
}

/// Whether an intrinsic row's argument category admits an argument, for
/// some admissible binding of an element parameter.
fn category_admits(category: &OperandCategory, argument: &ValueType) -> bool {
    match (category, argument) {
        (OperandCategory::Scalar(expected) | OperandCategory::Constant(expected), ValueType::Scalar(actual)) => {
            expected == actual
        }
        (
            OperandCategory::Readable {
                representation,
                rank,
            }
            | OperandCategory::Writable {
                representation,
                rank,
            },
            ValueType::Tensor(tensor),
        ) => {
            u32::try_from(tensor.rank()).ok() == Some(*rank)
                && match &tensor.elem {
                    Elem::Dtype(dtype) => crate::registry::dense(*dtype) == *representation,
                    Elem::Repr(actual) => actual == representation,
                    Elem::Param(_) => crate::registry::representation_info(*representation)
                        .decoded
                        .is_float(),
                }
        }
        (
            OperandCategory::Opaque { capability, name },
            ValueType::Opaque {
                capability: actual,
                name: actual_name,
            },
        ) => capability == actual && name == actual_name,
        _ => false,
    }
}

/// The element of a tensor of one representation.
fn representation_element(representation: crate::ids::RepresentationId) -> Elem {
    match crate::registry::representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => Elem::Dtype(dtype),
        RepresentationKind::Packed(_) | RepresentationKind::External(_) => Elem::Repr(representation),
    }
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
                                DiagnosticRule::Resolution,
                                callee.span,
                                "capability intrinsics use `<backend>.<capability>.<operation>`",
                            );
                            return None;
                        }
                    },
                    _ => {
                        self.error(
                            DiagnosticRule::Resolution,
                            callee.span,
                            "capability intrinsics use `<backend>.<capability>.<operation>`",
                        );
                        return None;
                    }
                };
                return self.intrinsic(backend, capability, name, args, span);
            }
            // L31: `index[B](w)`.
            A::Index { base, indices } if matches!(&base.kind, A::Name(n) if n.name == "index") => {
                return self.index_conversion(indices, bindings, args, span);
            }
            _ => {
                self.error(
                    DiagnosticRule::Type,
                    callee.span,
                    "only named functions can be called; functions and operations resolve statically",
                );
                return None;
            }
        };
        if self.lookup(&name.name).is_some() {
            self.error(
                DiagnosticRule::Type,
                name.span,
                format!("`{}` is a value, not a function", name.name),
            );
            return None;
        }
        let builtin = super::resolve::is_builtin_name(&name.name);
        if builtin && name.name != "repack" && !bindings.is_empty() {
            self.error(
                DiagnosticRule::Type,
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
        if let Some(op) = MathOp::parse(&name.name) {
            return self.math(op, &name.name, args, expected, span);
        }
        let positional = |c: &mut Checker, n: usize| -> bool {
            let ok = args.len() == n && args.iter().all(|a| a.name.is_none());
            if !ok {
                c.error(
                    DiagnosticRule::Type,
                    span,
                    format!("`{}` takes {n} positional argument(s)", name.name),
                );
            }
            ok
        };
        match name.name.as_str() {
            "index" => {
                self.error(
                    DiagnosticRule::Type,
                    span,
                    "`index` needs its bound: `index[B](w)`",
                );
                None
            }
            "range" => {
                self.error(
                    DiagnosticRule::Type,
                    span,
                    "`range` is a type, not a conversion; write `lo..hi`",
                );
                None
            }
            "repack" => {
                if !positional(self, 1) {
                    return None;
                }
                self.repack(bindings, &args[0].value, expected, span)
            }
            "to_owned" => {
                if !positional(self, 1) {
                    return None;
                }
                self.to_owned(&args[0].value, span)
            }
            "zeros_like" | "ones_like" => self.fill(&name.name, args, span),
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
                        DiagnosticRule::Type,
                        cond.span,
                        format!(
                            "the first operand of `select` is a mask (`bool`), found {}",
                            dtypes[0].name()
                        ),
                    );
                    return None;
                }
                let Some(target) = DType::promote(dtypes[1], dtypes[2]) else {
                    self.error(
                        DiagnosticRule::Type,
                        span,
                        format!(
                            "`select` between {} and {} needs an explicit cast",
                            dtypes[1].name(),
                            dtypes[2].name()
                        ),
                    );
                    return None;
                };
                let mut branches = self.promote_operands(vec![then, els], target).into_iter();
                let (then, els) = (branches.next()?, branches.next()?);
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
                    self.error(DiagnosticRule::Type, span, "`reduce(tile, axis, op)` takes three positional arguments and an optional `unordered=true|false`");
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
            "extent" => {
                if !positional(self, 2) {
                    return None;
                }
                let base = self.expr_inner(&args[0].value, None, true)?;
                let Some(s) = base.ty.shaped().cloned() else {
                    let shown = self.shown(&base.ty);
                    self.error(
                        DiagnosticRule::Type,
                        base.span,
                        format!("`extent` needs a tensor, view or tile, found {shown}"),
                    );
                    return None;
                };
                let axis = self.constant_axis(&args[1].value, s.rank())?;
                let sym = s.axes[axis];
                Some(self.primitive_expr(
                    PrimitiveId::Extent { axis: axis as u32 },
                    vec![base],
                    ValueType::Scalar(DType::I32),
                    Some(sym),
                    span,
                ))
            }
            "atomic" => self.atomic(args, span),
            _ => self.user_call(name, bindings, args, span),
        }
    }

    /// L31: `index[B](w)` converts a word (or quantity) proved in
    /// `0 <= w <= B - 1`.
    fn index_conversion(
        &mut self,
        indices: &[ast::Index],
        bindings: &[(ast::Ident, ast::Expr)],
        args: &[ast::Arg],
        span: Span,
    ) -> Option<CheckedExpr> {
        let ([ast::Index::Expr(bound)], [], [ast::Arg { name: None, value }]) = (indices, bindings, args)
        else {
            self.error(DiagnosticRule::Type, span, "`index[B](w)` takes one bound and one argument");
            return None;
        };
        let bound = self.expr(bound, Some(&ValueType::Integer))?;
        let bound = self.position_symbol(&bound)?;
        if !self.require_nonneg(DiagnosticRule::Type, bound, span, "an index bound may be negative") {
            return None;
        }
        let value = self.expr(value, None)?;
        if !matches!(
            value.ty,
            ValueType::Integer | ValueType::Index { .. } | ValueType::Scalar(DType::I32 | DType::U32)
        ) {
            let shown = self.shown(&value.ty);
            self.error(
                DiagnosticRule::Type,
                value.span,
                format!("`index[B](w)` converts an integer, found {shown}"),
            );
            return None;
        }
        self.index_position(value, bound, span)
    }

    fn repack(
        &mut self,
        bindings: &[(ast::Ident, ast::Expr)],
        source: &ast::Expr,
        expected: Option<&ValueType>,
        span: Span,
    ) -> Option<CheckedExpr> {
        let source = self.expr(source, None)?;
        let ValueType::Tensor(source_tensor) = source.ty.clone() else {
            self.error(DiagnosticRule::Type, source.span, "`repack` requires a tensor source");
            return None;
        };
        if matches!(source_tensor.elem, Elem::Dtype(_)) {
            self.error(
                DiagnosticRule::Type,
                source.span,
                "`repack` requires a registered external representation source",
            );
            return None;
        }
        let explicit = if bindings.is_empty() {
            None
        } else {
            let [(parameter, value)] = bindings else {
                self.error(
                    DiagnosticRule::Type,
                    span,
                    "`repack` accepts only the element binding `U = representation`",
                );
                return None;
            };
            if parameter.name != "U" {
                self.error(DiagnosticRule::Type, parameter.span, "`repack` destination binding is named `U`");
                return None;
            }
            let A::Name(representation) = &value.kind else {
                self.error(
                    DiagnosticRule::Type,
                    value.span,
                    "`repack` destination must name a representation",
                );
                return None;
            };
            if let Some(representation) = crate::registry::representation(&representation.name) {
                Some(RepresentationTarget::Concrete(representation))
            } else if self.sig.elem_params.contains(&representation.name) {
                Some(RepresentationTarget::Parameter(representation.name.clone()))
            } else {
                self.error(
                    DiagnosticRule::Resolution,
                    value.span,
                    format!(
                        "unknown representation or element parameter `{}`",
                        representation.name
                    ),
                );
                return None;
            }
        };
        let inferred = expected.and_then(|expected| match expected {
            ValueType::Tensor(tensor) => match &tensor.elem {
                Elem::Repr(representation) => Some(RepresentationTarget::Concrete(*representation)),
                Elem::Param(parameter) => Some(RepresentationTarget::Parameter(parameter.clone())),
                _ => None,
            },
            _ => None,
        });
        let Some(destination) = explicit.or(inferred) else {
            self.error(
                DiagnosticRule::Type,
                span,
                "`repack` needs `U = representation` or a representation-typed result context",
            );
            return None;
        };
        if let (Elem::Repr(source), RepresentationTarget::Concrete(destination)) =
            (&source_tensor.elem, &destination)
        {
            if crate::registry::representation_conversion(*source, *destination).is_none() {
                self.error(
                    DiagnosticRule::Type,
                    span,
                    format!(
                        "no exact representation conversion is registered from `{}` to `{}`",
                        crate::registry::representation_info(*source).name,
                        crate::registry::representation_info(*destination).name,
                    ),
                );
                return None;
            }
        }
        self.elements.conversion(
            &source_tensor.elem,
            match &destination {
                RepresentationTarget::Concrete(representation) => {
                    ElementTarget::Concrete(*representation)
                }
                RepresentationTarget::Parameter(parameter) => {
                    ElementTarget::Parameter(parameter.clone())
                }
            },
        );
        let destination_elem = match &destination {
            RepresentationTarget::Concrete(representation) => Elem::Repr(*representation),
            RepresentationTarget::Parameter(parameter) => Elem::Param(parameter.clone()),
        };
        let ty = ValueType::Tensor(TensorType::new(source_tensor.axes, destination_elem));
        Some(self.primitive_expr(
            PrimitiveId::RepresentationConvert(destination),
            vec![source],
            ty,
            None,
            span,
        ))
    }

    /// L22: whether a copied view selects whole packets on its packing axis:
    /// the axis is not selected, or a range whose start and length are
    /// multiples of the packet's logical group.
    fn packet_aligned(&mut self, value: &CheckedExpr, group: i64) -> bool {
        match &value.kind {
            CheckedExprKind::Primitive {
                id: PrimitiveId::SliceView { indices },
                operands,
                ..
            } => {
                let base = &operands[0];
                let Some(tensor) = base.ty.shaped() else {
                    return false;
                };
                let Some(packing) = tensor.packed_axis else {
                    return true;
                };
                let extent = tensor.axes[packing];
                let mut operand = 1;
                let mut aligned = true;
                for (axis, slot) in indices.iter().enumerate() {
                    match *slot {
                        intrinsics::IndexSlot::Point { .. } => {
                            aligned &= axis != packing;
                            operand += 1;
                        }
                        intrinsics::IndexSlot::Full => {}
                        intrinsics::IndexSlot::Range { start, end, .. } => {
                            let lo = start.then(|| {
                                let value = operands[operand].sym;
                                operand += 1;
                                value
                            });
                            let hi = end.then(|| {
                                let value = operands[operand].sym;
                                operand += 1;
                                value
                            });
                            if axis != packing {
                                continue;
                            }
                            let zero = self.arena.int(0);
                            let (Some(lo), Some(hi)) = (lo.unwrap_or(Some(zero)), hi.unwrap_or(Some(extent)))
                            else {
                                aligned = false;
                                continue;
                            };
                            let length = self.arena.int_sub(hi, lo);
                            let group = self.arena.int(group);
                            aligned &= super::prove::divide_exact(&mut self.arena, lo, group).is_some()
                                && super::prove::divide_exact(&mut self.arena, length, group).is_some();
                        }
                    }
                }
                aligned && self.packet_aligned(base, group)
            }
            CheckedExprKind::Primitive {
                id: PrimitiveId::Transpose | PrimitiveId::Reshape,
                operands,
                ..
            } => self.packet_aligned(&operands[0], group),
            _ => true,
        }
    }

    /// Whether a copied view selects its whole last axis (for an element
    /// parameter, whose packing is not known until binding).
    fn whole_last_axis(value: &CheckedExpr) -> bool {
        match &value.kind {
            CheckedExprKind::Primitive {
                id: PrimitiveId::SliceView { indices },
                operands,
                ..
            } => {
                let last = operands[0].ty.shaped().map_or(0, |t| t.rank()).saturating_sub(1);
                matches!(
                    indices.get(last),
                    Some(intrinsics::IndexSlot::Full)
                        | Some(intrinsics::IndexSlot::Range {
                            start: false,
                            end: false,
                            ..
                        })
                ) && Self::whole_last_axis(&operands[0])
            }
            CheckedExprKind::Primitive {
                id: PrimitiveId::Transpose | PrimitiveId::Reshape,
                ..
            } => false,
            _ => true,
        }
    }

    fn to_owned(&mut self, value: &ast::Expr, span: Span) -> Option<CheckedExpr> {
        let value = self.expr(value, None)?;
        // Ownership conversion is idempotent. An already-owned value needs
        // neither a diagnostic nor a second allocation.
        if matches!(self.class_of(&value), ValueClass::Owned) {
            return Some(value);
        }
        if !matches!(self.class_of(&value), ValueClass::Borrowed | ValueClass::Computed) {
            let shown = self.shown(&value.ty);
            self.error(
                DiagnosticRule::Type,
                value.span,
                format!("`to_owned` materializes a borrowed or computed tensor value, found {shown}"),
            );
            return None;
        }
        if !primitive(&PrimitiveId::Copy).accepts(&[value.ty.clone()]) {
            let shown = self.shown(&value.ty);
            self.error(
                DiagnosticRule::Type,
                value.span,
                format!("`to_owned` is not defined on {shown}"),
            );
            return None;
        }
        if let Some(tensor) = value.ty.shaped() {
            match &tensor.elem {
                Elem::Repr(representation) => {
                    let info = crate::registry::representation_info(*representation);
                    let group = match &info.kind {
                        RepresentationKind::Packed(layout) => Some(layout.group),
                        RepresentationKind::External(layout) => Some(layout.logical_group),
                        RepresentationKind::Dense(_) => None,
                    };
                    if let Some(group) = group {
                        if !self.packet_aligned(&value, i64::from(group)) {
                            self.error(
                                DiagnosticRule::Type,
                                value.span,
                                format!(
                                    "copying a `{}` view requires a packet-aligned selection on its packing axis",
                                    info.name
                                ),
                            );
                            return None;
                        }
                    }
                }
                Elem::Param(_) if !Self::whole_last_axis(&value) => {
                    let element = tensor.elem.clone();
                    self.elements.partial_copy(&element);
                }
                _ => {}
            }
        }
        let ty = value.ty.clone();
        Some(self.primitive_expr(PrimitiveId::Copy, vec![value], ty, None, span))
    }

    fn fill(&mut self, name: &str, args: &[ast::Arg], span: Span) -> Option<CheckedExpr> {
        let (Some(like), dtype) = (args.first().filter(|a| a.name.is_none()), args.get(1)) else {
            self.error(
                DiagnosticRule::Type,
                span,
                format!("`{name}(v, dtype=f32)` takes a shaped value and an optional dtype"),
            );
            return None;
        };
        if args.len() > 2 {
            self.error(
                DiagnosticRule::Type,
                span,
                format!("`{name}(v, dtype=f32)` takes a shaped value and an optional dtype"),
            );
            return None;
        }
        // Only the shape is used; the value is not read.
        let like = self.expr_inner(&like.value, None, true)?;
        let Some(s) = like.ty.shaped().cloned() else {
            let shown = self.shown(&like.ty);
            self.error(
                DiagnosticRule::Type,
                like.span,
                format!("`{name}` takes the shape of a tensor, view or tile, found {shown}"),
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
            None => Some(s.elem.read_dtype()),
        };
        let Some(dtype) = dtype else {
            self.error(DiagnosticRule::Type, span, "the second argument is `dtype=<dtype name>`");
            return None;
        };
        let ty = ValueType::Tensor(TensorType::new(s.axes, Elem::Dtype(dtype)));
        let value = if name == "zeros_like" {
            intrinsics::FillConstant::Zero
        } else {
            intrinsics::FillConstant::One
        };
        Some(self.primitive_expr(PrimitiveId::Fill(value), vec![like], ty, None, span))
    }

    fn constant_axis(&mut self, e: &ast::Expr, rank: usize) -> Option<usize> {
        let axis = match &e.kind {
            A::Int(v) => usize::try_from(*v).ok().filter(|a| *a < rank),
            _ => None,
        };
        if axis.is_none() {
            self.error(DiagnosticRule::Type, e.span, format!("the axis is a constant below rank {rank}"));
        }
        axis
    }

    fn math(
        &mut self,
        op: MathOp,
        name: &str,
        args: &[ast::Arg],
        expected: Option<&ValueType>,
        span: Span,
    ) -> Option<CheckedExpr> {
        let arity = op.arity();
        if args.len() != arity || args.iter().any(|a| a.name.is_some()) {
            self.error(
                DiagnosticRule::Type,
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
                        DiagnosticRule::Type,
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
                DiagnosticRule::Type,
                span,
                format!(
                    "`{name}` needs {} operands, found {}",
                    if float_only { "float" } else { "numeric" },
                    dtype.name()
                ),
            );
            return None;
        }
        let operands = self.promote_operands(out, dtype);
        self.elementwise_primitive(PrimitiveId::Math(op), operands, axes, span)
    }

    fn reduce(&mut self, args: &[ast::Arg], unordered: bool, span: Span) -> Option<CheckedExpr> {
        let t = self.expr(&args[0].value, None)?;
        let Some(s) = t.ty.shaped().cloned() else {
            let shown = self.shown(&t.ty);
            self.error(
                DiagnosticRule::Type,
                t.span,
                format!("`reduce` reduces a dense tensor value, found {shown}"),
            );
            return None;
        };
        // L9: a reduction folds dense numeric (non-bool) elements.
        let Some(dtype) = s.elem.dense_dtype() else {
            self.error(
                DiagnosticRule::Type,
                t.span,
                "`reduce` reduces a dense tile; decode packed values first",
            );
            return None;
        };
        if dtype == DType::Bool {
            self.error(
                DiagnosticRule::Type,
                t.span,
                "`reduce` folds numeric elements; convert a mask with `i32(mask)` first",
            );
            return None;
        }
        let axis = self.constant_axis(&args[1].value, s.rank())?;
        let op = match &args[2].value.kind {
            A::Name(n) => intrinsics::ReduceOp::parse(&n.name),
            _ => None,
        };
        let Some(op) = op else {
            self.error(
                DiagnosticRule::Type,
                args[2].value.span,
                "`reduce` needs an operation: sum, max, min or argmax",
            );
            return None;
        };
        if unordered && op == intrinsics::ReduceOp::Argmax {
            self.error(
                DiagnosticRule::Type,
                span,
                "`argmax` has one defined winner (ties go to the smaller index) and never accepts `unordered`",
            );
            return None;
        }
        self.elements.decoded_read(&s.elem);
        // §2.3.6 (2): an element-parameter operand reads as its decoded f32.
        let t = if matches!(s.elem, Elem::Param(_)) {
            self.promote_operands(vec![t], DType::F32).remove(0)
        } else {
            t
        };
        let check_nonempty = op != intrinsics::ReduceOp::Sum && {
            let one = self.arena.int(1);
            let slack = self.arena.int_sub(s.axes[axis], one);
            !super::prove::nonneg(&self.arena, &self.facts, slack)
        };
        let id = PrimitiveId::Reduce {
            op,
            axis: axis as u32,
            unordered,
            check_nonempty,
        };
        let Some(ty) = reduction_result(&t.ty, op, axis as u32) else {
            self.error(
                DiagnosticRule::Type,
                t.span,
                "`reduce` reduces a dense tile; decode packed values first",
            );
            return None;
        };
        Some(self.primitive_expr(id, vec![t], ty, None, span))
    }

    fn reshape(&mut self, args: &[ast::Arg], span: Span) -> Option<CheckedExpr> {
        let base = self.expr_inner(&args[0].value, None, true)?;
        let Some(s) = base.ty.shaped().cloned() else {
            let shown = self.shown(&base.ty);
            self.error(
                DiagnosticRule::Type,
                base.span,
                format!("`reshape` needs a tensor, view or tile, found {shown}"),
            );
            return None;
        };
        if matches!(s.elem, Elem::Repr(_)) {
            self.error(DiagnosticRule::Type, span, "`reshape` requires dense storage: packets run along the last axis of a packed value");
            return None;
        }
        let A::Tuple(dimensions) = &args[1].value.kind else {
            self.error(
                DiagnosticRule::Type,
                args[1].value.span,
                "the `reshape` target is a tuple of shape expressions",
            );
            return None;
        };
        let mut source = self.arena.int(1);
        for axis in &s.axes {
            source = self.arena.int_mul(source, *axis);
        }
        let mut axes = Vec::new();
        let mut operands = vec![base];
        let mut target = self.arena.int(1);
        for dimension in dimensions {
            let d = self.expr(dimension, Some(&ValueType::Integer))?;
            let extent = self.position_symbol(&d)?;
            if !self.require_nonneg(DiagnosticRule::Type, extent, d.span, "reshape extent may be negative") {
                return None;
            }
            target = self.arena.int_mul(target, extent);
            axes.push(extent);
            operands.push(d);
        }
        let difference = self.arena.int_sub(target, source);
        if axes.is_empty() || !super::prove::zero(&self.arena, &self.facts, difference) {
            self.error(
                DiagnosticRule::Type,
                span,
                "`reshape` must provably preserve element correspondence",
            );
            return None;
        }
        let ty = ValueType::Tensor(TensorType::new(axes, s.elem));
        Some(self.primitive_expr(PrimitiveId::Reshape, operands, ty, None, span))
    }

    /// `atomic(add|max|min, t[i], value)` (L12): the place is one element of
    /// a `&mut` tensor parameter or of a `let mut` tensor.
    fn atomic(&mut self, args: &[ast::Arg], span: Span) -> Option<CheckedExpr> {
        if !(args.len() == 3 && args.iter().all(|a| a.name.is_none())) {
            self.error(
                DiagnosticRule::Atomic,
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
                        DiagnosticRule::Atomic,
                        n.span,
                        format!("`atomic({}, …)`: the operation is add, max or min", n.name),
                    );
                    return None;
                }
            },
            _ => {
                self.error(
                    DiagnosticRule::Atomic,
                    args[0].value.span,
                    "`atomic` needs an operation name: add, max or min",
                );
                return None;
            }
        };
        let A::Index { base, .. } = &args[1].value.kind else {
            self.error(
                DiagnosticRule::Atomic,
                args[1].value.span,
                "the `atomic` place is an element of a `&mut` tensor or a `let mut` tensor: `atomic(add, t[i], v)`",
            );
            return None;
        };
        let (root, indices, selected) = self.place(&args[1].value)?;
        let ValueType::Scalar(dtype) = selected else {
            let shown = self.shown(&selected);
            self.error(
                DiagnosticRule::Atomic,
                args[1].value.span,
                format!("the `atomic` place selects one element, found {shown}"),
            );
            return None;
        };
        if !intrinsics::atomic_dtype(dtype) {
            self.error(
                DiagnosticRule::Atomic,
                span,
                format!(
                    "`atomic(add, …)` is defined for f32, f16, bf16, i32 and u32 elements, not {}",
                    dtype.name()
                ),
            );
            return None;
        }
        // Packed planes are readable but not writable.
        if let Some(shaped) = self.locals[root.index()].ty.shaped() {
            if matches!(shaped.elem, Elem::Repr(_)) {
                self.error(
                    DiagnosticRule::Type,
                    span,
                    "packed representations are readable and decodable but not writable",
                );
                return None;
            }
        }
        let value = self.expr(&args[2].value, Some(&ValueType::Scalar(dtype)))?;
        if value.ty.scalar_dtype() != Some(dtype) {
            let shown = self.shown(&value.ty);
            self.error(
                DiagnosticRule::Atomic,
                value.span,
                format!("the `atomic` value must be {}, found {shown}", dtype.name()),
            );
            return None;
        }
        let A::Name(name) = &base.kind else {
            unreachable!("a checked element place indexes a named tensor")
        };
        let binding = self.lookup(&name.name).expect("a checked place names a local");
        let storage_root = self.root_var_local(binding);
        if !self.writable_root(storage_root) || !self.writable_root(binding) {
            self.error(
                DiagnosticRule::Atomic,
                span,
                "an `atomic` place is an element of a `&mut tensor` parameter or of a `let mut` tensor",
            );
            return None;
        }
        if self
            .live_borrows()
            .iter()
            .any(|(borrow, borrowed, _)| borrowed.local == root && borrow.local != binding)
        {
            let name = self.locals[root.index()].name.clone();
            self.error(
                DiagnosticRule::Ownership,
                span,
                format!("cannot update `{name}` atomically while a tensor borrow is live"),
            );
            return None;
        }
        let participants = self
            .logical_parallel
            .iter()
            .filter(|(floor, _)| storage_root.index() < *floor)
            .map(|(_, binder)| *binder)
            .collect::<Vec<_>>();
        let outcome = match op {
            intrinsics::AtomicOp::Add if dtype.is_float() => {
                super::ir::CheckedAssociationOutcome::Reassociated { accumulator: dtype }
            }
            intrinsics::AtomicOp::Add | intrinsics::AtomicOp::Max | intrinsics::AtomicOp::Min => {
                super::ir::CheckedAssociationOutcome::Exact
            }
        };
        let authority = super::ir::AtomicCapability::checked(
            super::ir::Place::Element {
                root: super::ownership::LocalPlace::root(storage_root),
                indices: indices.clone(),
            },
            participants,
            outcome,
        );
        Some(CheckedExpr::new(
            CheckedExprKind::Atomic {
                op,
                place: super::ownership::LocalPlace::root(binding),
                indices,
                value: Box::new(value),
                authority,
            },
            ValueType::Void,
            None,
            span,
        ))
    }

    /// The storage root of a binding local, before following view aliases.
    pub(crate) fn root_var_local(&self, id: LocalId) -> LocalId {
        self.local_storage_root(id)
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
        let Some(backend_id) = crate::registry::BackendName::parse(&backend.name) else {
            self.error(
                DiagnosticRule::Resolution,
                backend.span,
                format!("`{}` is not a value or a backend namespace", backend.name),
            );
            return None;
        };
        let Some(capability_id) = crate::registry::capability(backend_id, &capability.name) else {
            self.error(
                DiagnosticRule::Resolution,
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
        let overload = crate::registry::intrinsic_overloads(capability_id, &name.name);
        if overload.is_empty() {
            self.error(
                DiagnosticRule::Resolution,
                name.span,
                format!(
                    "capability `{}.{}` has no intrinsic `{}`",
                    backend.name, capability.name, name.name
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
        let mut checked = Vec::new();
        for arg in args {
            if arg.name.is_some() {
                self.error(
                    DiagnosticRule::Type,
                    arg.value.span,
                    format!("`{}` takes positional arguments only", name.name),
                );
                return None;
            }
            checked.push(self.expr(&arg.value, None)?);
        }
        // C1-3: the rows compatible with some admissible binding.
        let rows = overload
            .iter()
            .copied()
            .filter(|row| {
                let signature = crate::registry::intrinsic_signature(*row);
                signature.arguments.len() == checked.len()
                    && signature
                        .arguments
                        .iter()
                        .zip(&checked)
                        .all(|(parameter, argument)| category_admits(&parameter.category, &argument.ty))
            })
            .collect::<Vec<_>>();
        let Some(first) = rows.first().copied() else {
            let types = checked
                .iter()
                .map(|argument| self.shown(&argument.ty))
                .collect::<Vec<_>>()
                .join(", ");
            self.error(
                DiagnosticRule::Type,
                span,
                format!("no row of `{}` takes arguments ({types})", name.name),
            );
            return None;
        };
        let signature = crate::registry::intrinsic_signature(first);
        let result = self.intrinsic_result(signature, &checked, span)?;
        for row in &rows[1..] {
            let other = self.intrinsic_result(crate::registry::intrinsic_signature(*row), &checked, span)?;
            if !self.same_ty(&result, &other) {
                self.error(
                    DiagnosticRule::Type,
                    span,
                    format!("the rows of `{}` admitted by these arguments disagree on the result type", name.name),
                );
                return None;
            }
        }
        for (parameter, argument) in signature.arguments.iter().zip(&checked) {
            if let ValueType::Tensor(tensor) = &argument.ty {
                match parameter.category {
                    OperandCategory::Writable { .. } => self.elements.stored(&tensor.elem),
                    _ => self.elements.decoded_read(&tensor.elem),
                }
            }
        }
        self.intrinsic_placement(signature, span)?;
        Some(CheckedExpr::new(
            CheckedExprKind::Intrinsic {
                overload: IntrinsicOverload {
                    capability: capability_id,
                    name: signature.name,
                    rows,
                },
                args: checked,
            },
            result,
            None,
            span,
        ))
    }

    /// L13: where a capability intrinsic may execute relative to the
    /// participants of the enclosing `parallel for`.
    fn intrinsic_placement(&mut self, signature: &IntrinsicSignature, span: Span) -> Option<()> {
        let local_parallel = !self.logical_parallel.is_empty();
        match signature.execution {
            IntrinsicExecution::WithinEnclosingParallel => {
                if !local_parallel {
                    self.placement.requires_enclosing_parallel = true;
                }
            }
            IntrinsicExecution::WholeTensor { .. } => {
                if local_parallel {
                    self.error(
                        DiagnosticRule::Placement,
                        span,
                        format!(
                            "whole-tensor intrinsic `{}` cannot appear under a `parallel for`",
                            signature.name
                        ),
                    );
                    return None;
                }
                self.placement.forbids_enclosing_parallel = true;
            }
        }
        if matches!(
            signature.effects.participation,
            IntrinsicParticipation::FullSubgroup
                | IntrinsicParticipation::FullWorkgroup
                | IntrinsicParticipation::FixedWorkgroup(_)
        ) {
            if !self.participant_uniform() {
                self.error(
                    DiagnosticRule::Placement,
                    span,
                    format!(
                        "cohort intrinsic `{}` needs every participant of its cohort; it is under control that differs between participants",
                        signature.name
                    ),
                );
                return None;
            }
            if !local_parallel {
                self.placement.requires_uniform_call_site = true;
            }
        }
        Some(())
    }

    /// The checked result type of one intrinsic row applied to `args`.
    fn intrinsic_result(
        &mut self,
        signature: &IntrinsicSignature,
        args: &[CheckedExpr],
        span: Span,
    ) -> Option<ValueType> {
        match &signature.result {
            IntrinsicResultType::Void => Some(ValueType::Void),
            IntrinsicResultType::Scalar(dtype) => Some(ValueType::Scalar(*dtype)),
            IntrinsicResultType::Opaque { capability, name } => Some(ValueType::Opaque {
                capability: *capability,
                name,
            }),
            IntrinsicResultType::Owned {
                representation,
                axes,
            } => {
                let projected = axes
                    .iter()
                    .map(|projection| {
                        args[projection.argument as usize]
                            .ty
                            .shaped()
                            .expect("a result projection selects a tensor argument")
                            .axes[projection.axis as usize]
                    })
                    .collect::<Vec<_>>();
                if let crate::registry::IntrinsicDenotation::MatrixProduct { accumulate } =
                    crate::registry::intrinsic_denotation(signature.id)
                {
                    let left = args[0].ty.shaped().expect("a matrix operand is a tensor");
                    let right = args[1].ty.shaped().expect("a matrix operand is a tensor");
                    if !self.same_extent(left.axes[1], right.axes[0]) {
                        let (left, right) = (self.render(left.axes[1]), self.render(right.axes[0]));
                        self.error(
                            DiagnosticRule::Type,
                            span,
                            format!("`{}` inner axes differ: `{left}` versus `{right}`", signature.name),
                        );
                        return None;
                    }
                    if accumulate {
                        let accumulator = args[2].ty.shaped().expect("a matrix accumulator is a tensor");
                        if accumulator
                            .axes
                            .iter()
                            .zip(&projected)
                            .any(|(actual, expected)| !self.same_extent(*actual, *expected))
                        {
                            self.error(
                                DiagnosticRule::Type,
                                args[2].span,
                                format!(
                                    "`{}` accumulator shape does not match the matrix product",
                                    signature.name
                                ),
                            );
                            return None;
                        }
                    }
                }
                Some(ValueType::Tensor(TensorType::new(
                    projected,
                    representation_element(*representation),
                )))
            }
        }
    }

    // ---- contract families ----

    /// Parameter ordinal -> argument ordinal (L3: one order for every member).
    fn arg_order(params: &[SigParam], args: &[ast::Arg]) -> Result<Vec<usize>, String> {
        if args.len() != params.len() {
            return Err(format!(
                "takes {} arguments, {} given",
                params.len(),
                args.len()
            ));
        }
        let mut order: Vec<Option<usize>> = vec![None; params.len()];
        let mut positional = 0;
        for (ordinal, arg) in args.iter().enumerate() {
            let slot = match &arg.name {
                None => {
                    positional += 1;
                    positional - 1
                }
                Some(label) => match params.iter().position(|p| p.name == label.name) {
                    Some(slot) => slot,
                    None => return Err(format!("has no parameter `{}`", label.name)),
                },
            };
            if order.get(slot).is_none_or(|bound| bound.is_some()) {
                return Err(format!("binds parameter {} twice", slot + 1));
            }
            order[slot] = Some(ordinal);
        }
        order
            .into_iter()
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| "leaves a parameter unbound".to_string())
    }

    /// Structure of one argument against its formal: kinds, ranks, tuple
    /// shapes and ownership admission. Dimensions are solved separately.
    fn admits_structure(&self, formal: &ValueType, actual: &ValueType) -> bool {
        match (formal, actual) {
            (ValueType::Scalar(a), ValueType::Scalar(b)) => a == b || (a.is_float() && b.is_float()),
            (ValueType::Index { .. }, ValueType::Index { .. } | ValueType::Integer)
            | (ValueType::Index { .. }, ValueType::Scalar(DType::I32 | DType::U32))
            | (ValueType::Range { .. }, ValueType::Range { .. }) => true,
            (ValueType::Tensor(p), ValueType::Tensor(a)) => {
                p.rank() == a.rank()
                    && (!matches!(a.elem, Elem::Repr(_))
                        || a.rank().checked_sub(1).is_some_and(|last| a.packed_axis == Some(last)))
            }
            (ValueType::Tuple(p), ValueType::Tuple(a)) => {
                p.len() == a.len() && p.iter().zip(a.iter()).all(|(p, a)| self.admits_structure(p, a))
            }
            (
                ValueType::Opaque {
                    capability: pc,
                    name: pn,
                },
                ValueType::Opaque {
                    capability: ac,
                    name: an,
                },
            ) => pc == ac && pn == an,
            _ => false,
        }
    }

    /// The contract of the family of `name` these arguments bind: its
    /// argument order and element bindings.
    fn select_family(
        &mut self,
        name: &ast::Ident,
        families: &[usize],
        ast_args: &[ast::Arg],
        args: &[CheckedExpr],
    ) -> Result<(usize, Vec<usize>, Elements), Vec<String>> {
        let resolved = self.env.resolved;
        let mut reasons = Vec::new();
        for family in families {
            let contract = resolved.families[*family].contract.index();
            let Some(Some(outcome)) = self.env.checked.get(contract) else {
                unreachable!("a callee is checked before its caller (L20)")
            };
            let params = &outcome.signature.params;
            let order = match Self::arg_order(params, ast_args) {
                Ok(order) => order,
                Err(reason) => {
                    reasons.push(format!("`{}` {reason}", name.name));
                    continue;
                }
            };
            let mut elements = Elements::default();
            let mut mismatch = None;
            for (parameter, ordinal) in params.iter().zip(&order) {
                let argument = &args[*ordinal];
                if !self.admits_structure(&parameter.ty, &argument.ty)
                    || !elements.unify_type(&parameter.ty, &argument.ty)
                {
                    mismatch = Some(format!(
                        "parameter `{}` expects {} but was given {}",
                        parameter.name,
                        parameter.ty.with_shapes(&outcome.arena, &|symbol| {
                            outcome
                                .signature
                                .dimension_of(symbol)
                                .map(|ordinal| outcome.signature.dimensions[ordinal].name.clone())
                                .unwrap_or_else(|| "?".to_owned())
                        }),
                        self.shown(&argument.ty)
                    ));
                    break;
                }
                if parameter.ownership == ParamOwnership::Exclusive
                    && !matches!(self.class_of(argument), ValueClass::Owned | ValueClass::Borrowed)
                {
                    mismatch = Some(format!(
                        "parameter `{}` requires a tensor place for exclusive access",
                        parameter.name
                    ));
                    break;
                }
            }
            match mismatch {
                Some(reason) => reasons.push(reason),
                None => return Ok((*family, order, elements)),
            }
        }
        Err(reasons)
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
            self.error(
                DiagnosticRule::Resolution,
                name.span,
                format!("`{}` is not declared", name.name),
            );
            return None;
        };
        // Seeds execute before the value arguments.
        let mut seeds = Vec::new();
        for (parameter, value) in bindings {
            let value = self.expr(value, Some(&ValueType::Integer))?;
            if !matches!(
                value.ty,
                ValueType::Integer | ValueType::Index { .. } | ValueType::Scalar(DType::I32 | DType::U32)
            ) {
                let shown = self.shown(&value.ty);
                self.error(
                    DiagnosticRule::Dimension,
                    value.span,
                    format!("a dimension binding is an integer, found {shown}"),
                );
                return None;
            }
            self.position_symbol(&value)?;
            seeds.push((parameter.clone(), value));
        }
        // Arguments are checked once, with scalar hints and write positions
        // from the first family whose contract the call can bind.
        let guide = families.iter().find_map(|family| {
            let contract = resolved.families[*family].contract.index();
            let outcome = self.env.checked.get(contract)?.as_ref()?;
            Self::arg_order(&outcome.signature.params, ast_args)
                .ok()
                .map(|order| (outcome, order))
        });
        let mut args = Vec::new();
        for (ordinal, arg) in ast_args.iter().enumerate() {
            let param = guide.as_ref().and_then(|(outcome, order)| {
                order
                    .iter()
                    .position(|o| *o == ordinal)
                    .map(|slot| &outcome.signature.params[slot])
            });
            let hint = param
                .and_then(|p| p.ty.scalar_dtype())
                .map(ValueType::Scalar);
            let write_only_candidate =
                param.is_some_and(|p| p.ownership == ParamOwnership::Exclusive);
            args.push(self.expr_inner(&arg.value, hint.as_ref(), write_only_candidate)?);
        }
        let (family, order, elements) = match self.select_family(name, families, ast_args, &args) {
            Ok(selected) => selected,
            Err(mut reasons) => {
                reasons.dedup();
                self.error(
                    DiagnosticRule::Type,
                    span,
                    format!(
                        "no definition of `{}` accepts these arguments: {}",
                        name.name,
                        reasons.join("; ")
                    ),
                );
                return None;
            }
        };
        let contract = resolved.families[family].contract.index();
        let Some(Some(outcome)) = self.env.checked.get(contract) else {
            unreachable!("a callee is checked before its caller (L20)")
        };
        // Cascade suppression: a contract whose own check failed has its
        // diagnostics; its call is not checked further.
        if !outcome.diagnostics.is_empty() {
            return None;
        }
        let plan = outcome
            .plan
            .as_ref()
            .expect("a family contract without diagnostics has its dimension plan");
        let dimension_symbols = outcome
            .signature
            .dimensions
            .iter()
            .map(|dimension| dimension.symbol)
            .collect::<Vec<_>>();
        let mut seeded = Vec::new();
        for (parameter, value) in &seeds {
            let Some(ordinal) = outcome
                .signature
                .dimensions
                .iter()
                .position(|dimension| dimension.name == parameter.name)
            else {
                self.error(
                    DiagnosticRule::Dimension,
                    parameter.span,
                    format!("`{}` has no dimension `{}`", name.name, parameter.name),
                );
                return None;
            };
            let value = value.sym.expect("a checked seed has its symbol");
            seeded.push((u32::try_from(ordinal).expect("dimension ordinal fits u32"), value));
        }
        // L16: the family plan solves every dimension from the actual axes.
        let actual_axes = |axis: &dimensions::ObservedAxis| {
            let mut ty = &args[order[axis.parameter as usize]].ty;
            for &index in &axis.path {
                let ValueType::Tuple(parts) = ty else {
                    unreachable!("an observed axis path selects tuple components")
                };
                ty = &parts.as_slice()[index as usize];
            }
            ty.shaped().expect("an observed axis is a tensor axis").axes[axis.axis as usize]
        };
        let values = match dimensions::apply_at_call(
            plan,
            &outcome.arena,
            &mut self.arena,
            &actual_axes,
            &seeded,
        ) {
            Ok(values) => values,
            Err(DimensionCallError::InexactDivision { dimension }) => {
                let dimension = outcome.signature.dimensions[dimension as usize].name.clone();
                self.error(
                    DiagnosticRule::Dimension,
                    span,
                    format!(
                        "dimension `{dimension}` of `{}` is not an exact quotient of these argument axes; bind it with `{}[{dimension} = …](…)`",
                        name.name, name.name
                    ),
                );
                return None;
            }
        };
        let seed_ordinals = seeded.iter().map(|(ordinal, _)| *ordinal).collect::<Vec<_>>();
        // Every formal axis equals its actual axis under the solved dimensions.
        let params = outcome.signature.params.clone();
        for (parameter_ordinal, parameter) in params.iter().enumerate() {
            let formal_axes = dimensions::tensor_axes(parameter_ordinal as u32, &parameter.ty);
            for (axis, formal) in formal_axes {
                let expected = substitute(&outcome.arena, &dimension_symbols, &values, &mut self.arena, formal);
                let actual = actual_axes(&axis);
                if !self.same_extent(expected, actual) {
                    let (expected, actual) = (self.render(expected), self.render(actual));
                    let seeded_here = super::prove::symbols(&outcome.arena, formal)
                        .into_iter()
                        .filter_map(|symbol| outcome.signature.dimension_of(symbol))
                        .any(|ordinal| seed_ordinals.contains(&(ordinal as u32)));
                    let cause = if seeded_here { "the dimension binding" } else { "the other arguments" };
                    self.error(
                        DiagnosticRule::Dimension,
                        args[order[parameter_ordinal]].span,
                        format!(
                            "argument `{}` of `{}` needs axis `{expected}` from {cause}, found `{actual}`",
                            parameter.name, name.name
                        ),
                    );
                    return None;
                }
            }
        }
        // Positivity and the contract's `where` at this call (L2).
        for (ordinal, dimension) in outcome.signature.dimensions.iter().enumerate() {
            let lower = self.arena.int(if dimension.admits_zero { 0 } else { 1 });
            let slack = self.arena.int_sub(values[ordinal], lower);
            if !super::prove::nonneg(&self.arena, &self.facts, slack) {
                let rendered = self.render(values[ordinal]);
                self.error(
                    DiagnosticRule::CallContract,
                    span,
                    format!(
                        "`{}` requires dimension `{}` = `{rendered}` to be at least {} at this call; guard the call",
                        name.name,
                        dimension.name,
                        if dimension.admits_zero { 0 } else { 1 }
                    ),
                );
                return None;
            }
        }
        for conjunct in outcome.signature.predicates.clone() {
            let expression = substitute(
                &outcome.arena,
                &dimension_symbols,
                &values,
                &mut self.arena,
                conjunct.predicate.expression(),
            );
            let proved = match conjunct.predicate {
                Predicate::NonNegative(_) => super::prove::nonneg(&self.arena, &self.facts, expression),
                Predicate::Zero(_) => super::prove::zero(&self.arena, &self.facts, expression),
                Predicate::NonZero(_) => self.facts.nonzero(&mut self.arena, expression),
            };
            if !proved {
                let file = resolved.declared[contract].file;
                let text = &self.env.texts[file]
                    [conjunct.span.start as usize..conjunct.span.end as usize];
                self.error(
                    DiagnosticRule::CallContract,
                    span,
                    format!("`{}` requires `{text}` at this call; guard the call", name.name),
                );
                return None;
            }
        }
        // L17, L31: index and range arguments by subsumption or position proof.
        for (parameter, ordinal) in params.iter().zip(&order) {
            let formal = substitute_type(
                &outcome.arena,
                &dimension_symbols,
                &values,
                &elements.arguments,
                &mut self.arena,
                &parameter.ty,
            );
            let argument = args[*ordinal].clone();
            let checked = match (&formal, &argument.ty) {
                (ValueType::Index { bound }, ValueType::Index { bound: actual }) => {
                    let slack = self.arena.int_sub(*bound, *actual);
                    if super::prove::nonneg(&self.arena, &self.facts, slack) {
                        argument
                    } else {
                        let span = argument.span;
                        self.index_position(argument, *bound, span)?
                    }
                }
                (ValueType::Index { bound }, _) => {
                    let span = argument.span;
                    self.index_position(argument, *bound, span)?
                }
                (ValueType::Range { bound }, ValueType::Range { bound: actual }) => {
                    let slack = self.arena.int_sub(*bound, *actual);
                    if !super::prove::nonneg(&self.arena, &self.facts, slack) {
                        let (formal, actual) = (self.render(*bound), self.render(*actual));
                        self.error(
                            DiagnosticRule::Type,
                            argument.span,
                            format!(
                                "parameter `{}` takes `range[{formal}]`; this range ends at `{actual}`; prove `{actual} <= {formal}` in scope",
                                parameter.name
                            ),
                        );
                        return None;
                    }
                    self.range_endpoints(argument, *bound)?
                }
                _ => argument,
            };
            args[*ordinal] = checked;
        }
        let result = substitute_type(
            &outcome.arena,
            &dimension_symbols,
            &values,
            &elements.arguments,
            &mut self.arena,
            &outcome.signature.result,
        );

        // Logical ownership is checked at the static call boundary.
        let mut arguments = Vec::new();
        for (parameter, ordinal) in params.iter().zip(&order) {
            super::ownership::argument_leaves(&parameter.ownership, &args[*ordinal], &mut arguments);
        }
        let mut accesses: Vec<(super::ownership::LocalPlace, ParamOwnership)> = Vec::new();
        let mut moves = std::collections::BTreeSet::new();
        for (ownership, argument) in &arguments {
            if *ownership == ParamOwnership::Value {
                continue;
            }
            if *ownership == ParamOwnership::Owned {
                let (_, taken) = match self.owned_consumption(argument) {
                    Ok(value) => value,
                    Err(error) => {
                        self.error(DiagnosticRule::Ownership, argument.span, error);
                        return None;
                    }
                };
                for place in taken {
                    if !moves.insert(place.clone()) {
                        self.error(
                            DiagnosticRule::Ownership,
                            argument.span,
                            "owned tensor leaf is moved into multiple arguments",
                        );
                        return None;
                    }
                }
            }
            let Some(root) = self.borrow_owner(argument) else {
                if matches!(ownership, ParamOwnership::Shared | ParamOwnership::Owned)
                    && matches!(
                        self.class_of(argument),
                        ValueClass::Computed | ValueClass::Owned
                    )
                {
                    continue;
                }
                self.error(
                    DiagnosticRule::Ownership,
                    argument.span,
                    "tensor parameter requires an actual tensor place",
                );
                return None;
            };
            if *ownership == ParamOwnership::Exclusive && !self.writable_place(&root) {
                self.error(
                    DiagnosticRule::Ownership,
                    argument.span,
                    "tensor place does not permit exclusive access",
                );
                return None;
            }
            if accesses.iter().any(|(previous, mode)| {
                previous.overlaps(&root)
                    && !(*mode == ParamOwnership::Shared && *ownership == ParamOwnership::Shared)
            }) {
                self.error(
                    DiagnosticRule::Ownership,
                    argument.span,
                    "overlapping tensor arguments cannot combine owned or exclusive access",
                );
                return None;
            }
            accesses.push((root, ownership.clone()));
        }
        for place in moves {
            self.consume_place(&place);
        }
        // Mutations of `&mut` parameters.
        for (mode, arg) in arguments {
            if mode != ParamOwnership::Exclusive {
                continue;
            }
            let base_local = |operands: &[CheckedExpr]| match operands.first().map(|o| &o.kind) {
                Some(CheckedExprKind::Local(v)) => Some(*v),
                _ => None,
            };
            let binding_local = match &arg.kind {
                CheckedExprKind::Local(v) => Some(*v),
                CheckedExprKind::Primitive {
                    id: PrimitiveId::SliceView { .. } | PrimitiveId::Reshape | PrimitiveId::Transpose,
                    operands,
                    ..
                } => base_local(operands),
                _ => self.value_place(&arg).map(|place| place.local),
            };
            let Some(binding_local) = binding_local else {
                self.error(
                    DiagnosticRule::Ownership,
                    arg.span,
                    "a `&mut` argument must name a tensor binding or a selection of one",
                );
                return None;
            };
            let storage_root = self.root_var_local(binding_local);
            self.write(storage_root, binding_local, arg.span)?;
        }

        // The members that apply at this call: the contract, then every
        // member whose elements unify and whose placement this call site
        // does not exclude. Initialization applicability is decided by the
        // initialization pass.
        let context = CallContext {
            enclosing_parallel: !self.logical_parallel.is_empty(),
            participant_uniform: self.participant_uniform(),
        };
        let callee_domain = outcome.element_domain.clone();
        let contract_arguments = elements
            .arguments
            .iter()
            .map(|(name, element)| (name.clone(), element.clone()))
            .collect::<Vec<_>>();
        self.elements.call(&callee_domain, &contract_arguments);
        let members = &resolved.families[family];
        let mut candidates = Vec::new();
        for member in std::iter::once(members.contract)
            .chain(members.bodies.iter().copied().filter(|body| *body != members.contract))
            .chain(members.lowerings.iter().copied())
        {
            let Some(Some(member_outcome)) = self.env.checked.get(member.index()) else {
                unreachable!("a family member is checked before its callers (L20)")
            };
            let mut member_elements = Elements::default();
            let unified = member_outcome
                .signature
                .params
                .iter()
                .zip(&order)
                .all(|(parameter, ordinal)| member_elements.unify_type(&parameter.ty, &args[*ordinal].ty));
            if !unified {
                continue;
            }
            if member != members.contract && member_outcome.placement.excluded_by(context) {
                continue;
            }
            let mut elem_args = member_elements
                .arguments
                .into_iter()
                .collect::<Vec<_>>();
            elem_args.sort_by(|a, b| a.0.cmp(&b.0));
            candidates.push(Candidate {
                definition: member,
                elem_args,
                requires_elems: member_elements.requires,
            });
        }
        let seeds = seeds
            .into_iter()
            .zip(&seeded)
            .map(|((_, value), (ordinal, _))| (*ordinal, value))
            .collect();
        let call = CheckedCall {
            family: resolved.declared[contract].family,
            seeds,
            arg_order: order,
            dimensions: values,
            context,
            candidates,
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

    /// A `range[bound]` argument built as `lo..hi` from words: each word
    /// endpoint is marked at its position (L31); the endpoints were proved in
    /// `0 <= lo <= hi` and the bound by subsumption.
    fn range_endpoints(&mut self, argument: CheckedExpr, bound: IntExpr) -> Option<CheckedExpr> {
        let CheckedExpr {
            kind:
                CheckedExprKind::Primitive {
                    id: PrimitiveId::RangeMake,
                    operands,
                    failure,
                },
            ty,
            sym,
            span,
        } = argument
        else {
            return Some(argument);
        };
        let one = self.arena.int(1);
        let past = self.arena.int_add(bound, one);
        let mut marked = Vec::new();
        for endpoint in operands {
            if matches!(endpoint.ty, ValueType::Scalar(DType::I32 | DType::U32)) {
                let span = endpoint.span;
                // An endpoint lies in `0..=bound`: an index of `bound + 1`.
                marked.push(self.index_position(endpoint, past, span)?);
            } else {
                marked.push(endpoint);
            }
        }
        Some(CheckedExpr::new(
            CheckedExprKind::Primitive {
                id: PrimitiveId::RangeMake,
                operands: marked,
                failure,
            },
            ty,
            sym,
            span,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::checked::{check_source, SourceFile, SourceSet};

    #[test]
    fn retired_spellings_are_unknown_names() {
        for spelling in ["exp_fast", "load", "clone", "decode"] {
            let error = check_source(SourceSet::new(vec![SourceFile {
                path: "retired.seismic".into(),
                text: format!(
                    "fn probe[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return {spelling}(x)\n"
                ),
            }]))
            .expect_err("a retired spelling is not a builtin");
            assert!(
                error.to_string().contains(&format!("`{spelling}` is not declared")),
                "{spelling}: {error}"
            );
        }
    }
}
