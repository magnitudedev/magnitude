//! Declarations: signatures, `where` predicates, contract families, lowering attachment
//! and explicit target coverage. One flat global namespace.

use super::ir::{DefKind, Ownership, Predicate, WhereConjunct};
use crate::checked::DiagnosticRule;
use crate::expr::{AnyExpr, ExprArena, IntExpr, SymbolId};
use crate::ids::{CapabilityId, FamilyId, FunctionId, ProgramId};
use crate::registry::{self, BackendName};
use crate::span::{Diagnostic, Span};
use crate::syntax::ast::{self, BinaryOp, ExprKind as A, ShapedHead, TypeKind};
use crate::types::{DType, Elem, NonEmpty, TensorType, ValueType};
use std::collections::{BTreeSet, HashMap};

/// A diagnostic attributed to a source file (index into the compiled file list).
#[derive(Clone, Debug)]
pub(crate) struct Located {
    pub file: usize,
    pub diagnostic: Diagnostic,
}

#[derive(Clone, Debug)]
pub(crate) struct SigParam {
    pub name: String,
    pub ownership: Ownership,
    pub ty: ValueType,
    pub span: Span,
}

/// One shape parameter of a declared signature. The only owner of positivity.
#[derive(Clone, Debug)]
pub(crate) struct SignatureDimension {
    pub name: String,
    pub symbol: SymbolId,
    /// `true` iff some `where` conjunct is `NonNegative(e)` with `prove::same(e, dim)`.
    pub admits_zero: bool,
}

/// A declaration's checked interface. Shapes are symbolic extents over its own
/// shape parameters.
#[derive(Debug)]
pub(crate) struct Sig {
    pub name: String,
    pub dimensions: Vec<SignatureDimension>,
    pub elem_params: Vec<String>,
    pub params: Vec<SigParam>,
    pub result: ValueType,
    pub predicates: Vec<WhereConjunct>,
    pub arena: ExprArena,
}

/// A declaration's signature rebuilt in the arena of its checked body.
#[derive(Debug)]
pub(crate) struct BodySig {
    pub name: String,
    pub dimensions: Vec<SignatureDimension>,
    pub elem_params: Vec<String>,
    pub params: Vec<SigParam>,
    pub result: ValueType,
    pub predicates: Vec<WhereConjunct>,
}

impl BodySig {
    /// The ordinal of a dimension symbol of this signature.
    pub(crate) fn dimension_of(&self, symbol: SymbolId) -> Option<usize> {
        self.dimensions
            .iter()
            .position(|dimension| dimension.symbol == symbol)
    }
}

impl Sig {
    /// Rebuild this immutable declaration signature into the definition arena
    /// that will also own its checked body. No expression handle is copied
    /// across arenas.
    pub(crate) fn for_body(&self) -> (BodySig, ExprArena) {
        let mut arena = ExprArena::new();
        let shape_symbols: Vec<_> = self
            .dimensions
            .iter()
            .enumerate()
            .map(|(ordinal, _)| {
                arena
                    .template_dimension(
                        u32::try_from(ordinal)
                            .expect("signature has more than u32::MAX dimensions"),
                    )
                    .0
            })
            .collect();
        let mut map = |symbol: SymbolId, destination: &mut ExprArena| {
            let ordinal = self
                .dimensions
                .iter()
                .position(|dimension| dimension.symbol == symbol)
                .unwrap_or_else(|| panic!("signature expression mentions a non-dimension symbol"));
            AnyExpr::Int(destination.int_symbol(shape_symbols[ordinal]))
        };
        let params = self
            .params
            .iter()
            .map(|parameter| SigParam {
                name: parameter.name.clone(),
                ownership: parameter.ownership.clone(),
                ty: transfer_type(&self.arena, &parameter.ty, &mut arena, &mut map),
                span: parameter.span,
            })
            .collect();
        let result = transfer_type(&self.arena, &self.result, &mut arena, &mut map);
        let predicates = self
            .predicates
            .iter()
            .map(|conjunct| {
                let mut transfer =
                    |value| super::xfer::transfer_int(&self.arena, value, &mut arena, &mut map);
                WhereConjunct {
                    predicate: match conjunct.predicate {
                        Predicate::NonNegative(value) => Predicate::NonNegative(transfer(value)),
                        Predicate::Zero(value) => Predicate::Zero(transfer(value)),
                        Predicate::NonZero(value) => Predicate::NonZero(transfer(value)),
                    },
                    span: conjunct.span,
                }
            })
            .collect();
        (
            BodySig {
                name: self.name.clone(),
                dimensions: self
                    .dimensions
                    .iter()
                    .zip(&shape_symbols)
                    .map(|(dimension, symbol)| SignatureDimension {
                        name: dimension.name.clone(),
                        symbol: *symbol,
                        admits_zero: dimension.admits_zero,
                    })
                    .collect(),
                elem_params: self.elem_params.clone(),
                params,
                result,
                predicates,
            },
            arena,
        )
    }
}

fn transfer_type(
    source: &ExprArena,
    ty: &ValueType,
    destination: &mut ExprArena,
    map: &mut super::xfer::SymbolMap<'_>,
) -> ValueType {
    match ty {
        ValueType::Scalar(dtype) => ValueType::Scalar(*dtype),
        ValueType::Integer => ValueType::Integer,
        ValueType::Index { bound } => ValueType::Index {
            bound: super::xfer::transfer_int(source, *bound, destination, map),
        },
        ValueType::Range { bound } => ValueType::Range {
            bound: super::xfer::transfer_int(source, *bound, destination, map),
        },
        ValueType::Tensor(tensor) => ValueType::Tensor(TensorType {
            axes: tensor
                .axes
                .iter()
                .map(|axis| super::xfer::transfer_int(source, *axis, destination, map))
                .collect(),
            elem: tensor.elem.clone(),
            packed_axis: tensor.packed_axis,
        }),
        ValueType::Tuple(items) => ValueType::Tuple(
            NonEmpty::new(
                items
                    .iter()
                    .map(|item| transfer_type(source, item, destination, map))
                    .collect(),
            )
            .expect("checked tuple is nonempty"),
        ),
        ValueType::Opaque { capability, name } => ValueType::Opaque {
            capability: *capability,
            name,
        },
        ValueType::Void => ValueType::Void,
    }
}

/// One definition before its body is checked.
pub(crate) struct Declared<'a> {
    pub sig: Sig,
    pub kind: DefKind,
    pub requires: Vec<(CapabilityId, Span)>,
    pub family: FamilyId,
    pub elem_bindings: Vec<(String, Elem)>,
    pub body: &'a ast::Block,
    pub file: usize,
    pub span: Span,
    pub name_span: Span,
}

fn requirements(
    paths: &[ast::CapabilityPath],
    target: Option<&str>,
) -> Result<Vec<(CapabilityId, Span)>, Vec<Diagnostic>> {
    let mut out = Vec::new();
    let mut diagnostics = Vec::new();
    for path in paths {
        let Some(backend) = BackendName::parse(&path.backend.name) else {
            diagnostics.push(Diagnostic::with_rule(
                DiagnosticRule::Resolution,
                path.backend.span,
                format!("`{}` is not a known target", path.backend.name),
            ));
            continue;
        };
        let Some(id) = registry::capability(backend, &path.capability.name) else {
            diagnostics.push(Diagnostic::with_rule(
                DiagnosticRule::Resolution,
                path.span,
                format!(
                    "`{}.{}` is not a known capability namespace",
                    path.backend.name, path.capability.name
                ),
            ));
            continue;
        };
        match target {
            None => diagnostics.push(Diagnostic::with_rule(
DiagnosticRule::Capability,
                path.span,
                format!(
                    "portable functions cannot require backend capability `{}`",
                    format!("{}.{}", path.backend.name, path.capability.name)
                ),
            )),
            Some(backend) if backend != path.backend.name => diagnostics.push(Diagnostic::with_rule(
DiagnosticRule::Capability,
                path.span,
                format!(
                    "capability `{}` belongs to backend `{}`, but this declaration is for `{backend}`",
                    format!("{}.{}", path.backend.name, path.capability.name), path.backend.name
                ),
            )),
            Some(_) if out.iter().any(|(existing, _)| existing == &id) => diagnostics.push(
                Diagnostic::with_rule(DiagnosticRule::Capability, path.span, format!("capability `{}.{}` is required more than once", path.backend.name, path.capability.name)),
            ),
            Some(_) => out.push((id, path.span)),
        }
    }
    if diagnostics.is_empty() {
        Ok(out)
    } else {
        Err(diagnostics)
    }
}

/// The members of one contract family, as resolution attaches them.
#[derive(Clone, Debug)]
pub(crate) struct DeclaredFamily {
    pub name: String,
    /// The first declared portable body: the family's meaning (L2).
    pub contract: FunctionId,
    pub bodies: Vec<FunctionId>,
    pub lowerings: Vec<FunctionId>,
}

pub(crate) struct Resolved<'a> {
    pub declared: Vec<Declared<'a>>,
    pub families: Vec<DeclaredFamily>,
    /// Function name -> indices into `families`.
    pub by_name: HashMap<String, Vec<usize>>,
    pub call_graph: NameCallGraph,
}

/// Names a call dispatches to a builtin meaning before any user family is
/// looked up, plus the quantity type names `index` and `range` (L31). A
/// definition cannot take one, so no call silently reaches a builtin instead of
/// the definition. The one list: `check/call.rs` dispatches exactly these.
pub(crate) fn is_builtin_name(name: &str) -> bool {
    DType::from_name(name).is_some()
        || crate::intrinsics::MathOp::parse(name).is_some()
        || matches!(
            name,
            "to_owned"
                | "zeros_like"
                | "ones_like"
                | "select"
                | "reduce"
                | "reshape"
                | "extent"
                | "atomic"
                | "repack"
                | "index"
                | "range"
        )
}

/// The name-level call graph: every definition bearing a called name is a
/// callee of the caller. Recursion is rejected on this graph.
pub(crate) struct NameCallGraph {
    /// Definition -> definitions it may call.
    callees: Vec<BTreeSet<usize>>,
}

/// Definitions forming a call cycle, in call order.
pub(crate) struct RecursionCycle {
    definitions: Vec<usize>,
}

impl NameCallGraph {
    fn new(
        declared: &[Declared<'_>],
        families: &[DeclaredFamily],
        by_name: &HashMap<String, Vec<usize>>,
    ) -> Self {
        let callees = declared
            .iter()
            .map(|definition| {
                let mut names = BTreeSet::new();
                called_names(definition.body, &mut names);
                names
                    .iter()
                    .filter_map(|name| by_name.get(name))
                    .flatten()
                    .flat_map(|&family| {
                        families[family]
                            .bodies
                            .iter()
                            .chain(&families[family].lowerings)
                            .map(|member| member.index())
                    })
                    .collect()
            })
            .collect();
        Self { callees }
    }

    /// Every definition after all of its callees.
    pub(crate) fn bottom_up_order(&self) -> Result<Vec<usize>, RecursionCycle> {
        fn visit(
            graph: &NameCallGraph,
            definition: usize,
            state: &mut [u8],
            stack: &mut Vec<usize>,
            order: &mut Vec<usize>,
        ) -> Result<(), RecursionCycle> {
            match state[definition] {
                2 => return Ok(()),
                1 => {
                    let start = stack
                        .iter()
                        .position(|&member| member == definition)
                        .expect("a definition on the visit stack is in progress");
                    return Err(RecursionCycle {
                        definitions: stack[start..].to_vec(),
                    });
                }
                _ => {}
            }
            state[definition] = 1;
            stack.push(definition);
            for &callee in &graph.callees[definition] {
                visit(graph, callee, state, stack, order)?;
            }
            stack.pop();
            state[definition] = 2;
            order.push(definition);
            Ok(())
        }
        let mut state = vec![0u8; self.callees.len()];
        let mut stack = Vec::new();
        let mut order = Vec::with_capacity(self.callees.len());
        for definition in 0..self.callees.len() {
            visit(self, definition, &mut state, &mut stack, &mut order)?;
        }
        Ok(order)
    }
}

impl RecursionCycle {
    pub(crate) fn diagnostic(&self, declared: &[Declared<'_>]) -> Located {
        let first = &declared[self.definitions[0]];
        let chain = self
            .definitions
            .iter()
            .chain(&self.definitions[..1])
            .map(|&definition| format!("`{}`", declared[definition].sig.name))
            .collect::<Vec<_>>()
            .join(" -> ");
        Located {
            file: first.file,
            diagnostic: Diagnostic::with_rule(
                DiagnosticRule::Recursion,
                first.name_span,
                format!("recursive call chain {chain}: recursion is rejected"),
            ),
        }
    }
}

/// Names called by a body, including calls nested in arguments and shape bindings.
fn called_names(block: &ast::Block, out: &mut BTreeSet<String>) {
    fn index(index: &ast::Index, out: &mut BTreeSet<String>) {
        match index {
            ast::Index::Expr(expression) => expr(expression, out),
            ast::Index::Slice { start, end } => {
                start.iter().chain(end).for_each(|bound| expr(bound, out));
            }
        }
    }

    fn expr(expression: &ast::Expr, out: &mut BTreeSet<String>) {
        match &expression.kind {
            ast::ExprKind::Call {
                callee,
                bindings,
                args,
            } => {
                match &callee.kind {
                    ast::ExprKind::Name(name) => {
                        out.insert(name.name.clone());
                    }
                    _ => expr(callee, out),
                }
                bindings.iter().for_each(|(_, value)| expr(value, out));
                args.iter().for_each(|argument| expr(&argument.value, out));
            }
            ast::ExprKind::Tuple(items) => items.iter().for_each(|item| expr(item, out)),
            ast::ExprKind::Range { lo, hi } => {
                expr(lo, out);
                expr(hi, out);
            }
            ast::ExprKind::Tensor { shape, .. } => shape.iter().for_each(|axis| expr(axis, out)),
            ast::ExprKind::Index { base, indices } => {
                expr(base, out);
                indices.iter().for_each(|item| index(item, out));
            }
            ast::ExprKind::Attr { base, .. } => expr(base, out),
            ast::ExprKind::Unary { expr: operand, .. } => expr(operand, out),
            ast::ExprKind::Binary { lhs, rhs, .. } => {
                expr(lhs, out);
                expr(rhs, out);
            }
            ast::ExprKind::Int(_)
            | ast::ExprKind::Float(_)
            | ast::ExprKind::Inf
            | ast::ExprKind::Bool(_)
            | ast::ExprKind::Name(_) => {}
        }
    }

    for statement in &block.stmts {
        match &statement.kind {
            ast::StmtKind::Let { value, .. } => expr(value, out),
            ast::StmtKind::Assign { target, value, .. } => {
                expr(target, out);
                expr(value, out);
            }
            ast::StmtKind::For { iter, body, .. } => {
                expr(iter, out);
                called_names(body, out);
            }
            ast::StmtKind::If { cond, then, els } => {
                expr(cond, out);
                called_names(then, out);
                if let Some(els) = els {
                    called_names(els, out);
                }
            }
            ast::StmtKind::Return(values) => values.iter().for_each(|value| expr(value, out)),
            ast::StmtKind::Expr(expression) => expr(expression, out),
        }
    }
}

/// A shape expression: integers, shape parameters, and `+ - * / %` over them.
pub(crate) fn shape_expr(
    e: &ast::Expr,
    shape_params: &[String],
    shape_symbols: &[SymbolId],
    arena: &mut ExprArena,
) -> Result<IntExpr, Diagnostic> {
    match &e.kind {
        A::Int(v) => i64::try_from(*v)
            .map(|value| arena.int(value))
            .map_err(|_| {
                Diagnostic::with_rule(
                    DiagnosticRule::Type,
                    e.span,
                    "shape constant does not fit a signed 64-bit integer",
                )
            }),
        A::Name(n) if shape_params.contains(&n.name) => {
            let ordinal = shape_params
                .iter()
                .position(|name| name == &n.name)
                .expect("shape parameter lookup disagrees with contains");
            Ok(arena.int_symbol(shape_symbols[ordinal]))
        }
        A::Name(n) => Err(Diagnostic::with_rule(
            DiagnosticRule::Resolution,
            n.span,
            format!("`{}` is not a declared shape parameter", n.name),
        )),
        A::Binary { op, lhs, rhs } => {
            let l = shape_expr(lhs, shape_params, shape_symbols, arena)?;
            let r = shape_expr(rhs, shape_params, shape_symbols, arena)?;
            return match op {
                BinaryOp::Add => Ok(arena.int_add(l, r)),
                BinaryOp::Sub => Ok(arena.int_sub(l, r)),
                BinaryOp::Mul => Ok(arena.int_mul(l, r)),
                BinaryOp::Div | BinaryOp::Rem => {
                    if super::prove::constant(arena, r).is_some_and(|c| c <= 0) {
                        return Err(Diagnostic::with_rule(
                            DiagnosticRule::Dimension,
                            rhs.span,
                            "shape divisor must be positive",
                        ));
                    }
                    Ok(if *op == BinaryOp::Div {
                        arena.int_div(l, r)
                    } else {
                        arena.int_rem(l, r)
                    })
                }
                _ => Err(Diagnostic::with_rule(
                    DiagnosticRule::Type,
                    e.span,
                    "only + - * / % are allowed in shapes",
                )),
            };
        }
        _ => Err(Diagnostic::with_rule(
            DiagnosticRule::Type,
            e.span,
            "a shape is an integer expression over shape parameters",
        )),
    }
}

/// Element descriptor: dtype, representation, or an implicit element parameter (capitalized name).
pub(crate) fn elem_of(
    name: &ast::Ident,
    elem_params: &mut Vec<String>,
) -> Result<Elem, Diagnostic> {
    if let Some(d) = DType::from_name(&name.name) {
        return Ok(Elem::Dtype(d));
    }
    if let Some(representation) = registry::representation(&name.name) {
        return Ok(Elem::Repr(representation));
    }
    if name
        .name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase())
    {
        if !elem_params.contains(&name.name) {
            elem_params.push(name.name.clone());
        }
        return Ok(Elem::Param(name.name.clone()));
    }
    Err(Diagnostic::with_rule(
        DiagnosticRule::Resolution,
        name.span,
        format!(
            "`{}` is not a dtype, a representation, or an element parameter",
            name.name
        ),
    ))
}

fn type_from_ast(
    t: &ast::TypeExpr,
    shape_params: &[String],
    shape_symbols: &[SymbolId],
    arena: &mut ExprArena,
    elem_params: &mut Vec<String>,
) -> Result<ValueType, Diagnostic> {
    match &t.kind {
        TypeKind::Scalar(name) => match DType::from_name(&name.name) {
            Some(d) => Ok(ValueType::Scalar(d)),
            None if name
                .name
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase()) =>
            {
                Err(Diagnostic::with_rule(
DiagnosticRule::Type,
                    name.span,
                    format!(
                        "element parameter `{}` cannot be a scalar type: scalar values have a concrete dtype",
                        name.name
                    ),
                ))
            }
            None => Err(Diagnostic::with_rule(
DiagnosticRule::Resolution,
                name.span,
                format!("unknown type `{}`", name.name),
            )),
        },
        TypeKind::Index(bound) => Ok(ValueType::Index {
            bound: shape_expr(bound, shape_params, shape_symbols, arena)?,
        }),
        TypeKind::Range(bound) => Ok(ValueType::Range {
            bound: shape_expr(bound, shape_params, shape_symbols, arena)?,
        }),
        TypeKind::Shaped { head, shape, elem } => {
            if shape.is_empty() {
                return Err(Diagnostic::with_rule(DiagnosticRule::Type, t.span, "a tensor type needs a shape"));
            }
            let mut axes = Vec::new();
            for e in shape {
                axes.push(shape_expr(e, shape_params, shape_symbols, arena)?);
            }
            let shaped = TensorType::new(axes, elem_of(elem, elem_params)?);
            let _ = head; // ownership is recorded beside the type, never in it
            Ok(ValueType::Tensor(shaped))
        }
        TypeKind::Tuple(items) => {
            let mut out = Vec::new();
            for item in items.iter() {
                let ty = type_from_ast(item, shape_params, shape_symbols, arena, elem_params)?;
                if ty.is_void() {
                    return Err(Diagnostic::with_rule(
DiagnosticRule::Type,
                        item.span,
                        "`void` is not a tuple component",
                    ));
                }
                out.push(ty);
            }
            // An empty tuple component list canonicalizes to `Void`.
            Ok(NonEmpty::new(out)
                .map(ValueType::Tuple)
                .unwrap_or(ValueType::Void))
        }
        TypeKind::Void => Ok(ValueType::Void),
    }
}

/// Conjuncts of a `where` clause: comparisons, divisibility and equalities over
/// shape parameters and integer literals.
fn predicates_of(
    e: &ast::Expr,
    shape_params: &[String],
    shape_symbols: &[SymbolId],
    arena: &mut ExprArena,
    out: &mut Vec<WhereConjunct>,
) -> Result<(), Diagnostic> {
    match &e.kind {
        A::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
        } => {
            predicates_of(lhs, shape_params, shape_symbols, arena, out)?;
            predicates_of(rhs, shape_params, shape_symbols, arena, out)
        }
        A::Binary { op, lhs, rhs } => {
            let l = shape_expr(lhs, shape_params, shape_symbols, arena)?;
            let r = shape_expr(rhs, shape_params, shape_symbols, arena)?;
            let one = arena.int(1);
            let predicate = match op {
                BinaryOp::Ge => Predicate::NonNegative(arena.int_sub(l, r)),
                BinaryOp::Gt => { let d = arena.int_sub(l, r); Predicate::NonNegative(arena.int_sub(d, one)) },
                BinaryOp::Le => Predicate::NonNegative(arena.int_sub(r, l)),
                BinaryOp::Lt => { let d = arena.int_sub(r, l); Predicate::NonNegative(arena.int_sub(d, one)) },
                BinaryOp::Eq => Predicate::Zero(arena.int_sub(l, r)),
                BinaryOp::Ne => Predicate::NonZero(arena.int_sub(l, r)),
                _ => return Err(Diagnostic::with_rule(DiagnosticRule::Type, e.span, "a `where` predicate is a conjunction of comparisons, divisibility and equalities over shape parameters")),
            };
            out.push(WhereConjunct {
                predicate,
                span: e.span,
            });
            Ok(())
        }
        _ => Err(Diagnostic::with_rule(
DiagnosticRule::Type,
            e.span,
            "a `where` predicate is a conjunction of comparisons, divisibility and equalities over shape parameters",
        )),
    }
}

pub(crate) fn signature_of(
    name: &str,
    s: &ast::Signature,
    extra_predicates: &[ast::Expr],
    body: &ast::Block,
) -> Result<Sig, Diagnostic> {
    let mut arena = ExprArena::new();
    let mut shape_params: Vec<String> = Vec::new();
    for p in &s.shape {
        if shape_params.contains(&p.name) {
            return Err(Diagnostic::with_rule(
                DiagnosticRule::Resolution,
                p.span,
                format!("duplicate shape parameter `{}`", p.name),
            ));
        }
        shape_params.push(p.name.clone());
    }
    // Definition-local shape symbols are lexical binders. They are transferred
    // into call dimensions only when an exported entry is monomorphized.
    let mut shape_symbols = Vec::with_capacity(shape_params.len());
    for _ in &shape_params {
        let (symbol, _) = arena.template_dimension(
            u32::try_from(shape_symbols.len())
                .expect("signature has more than u32::MAX dimensions"),
        );
        shape_symbols.push(symbol);
    }
    let mut elem_params = Vec::new();
    let mut params: Vec<SigParam> = Vec::new();
    for p in &s.params {
        if params.iter().any(|q| q.name == p.name.name) || shape_params.contains(&p.name.name) {
            return Err(Diagnostic::with_rule(
                DiagnosticRule::Resolution,
                p.name.span,
                format!("duplicate parameter `{}`", p.name.name),
            ));
        }
        let ty = type_from_ast(
            &p.ty,
            &shape_params,
            &shape_symbols,
            &mut arena,
            &mut elem_params,
        )?;
        reject_rank_zero_packed(&ty, p.ty.span)?;
        if ty.is_void() {
            return Err(Diagnostic::with_rule(
                DiagnosticRule::Type,
                p.ty.span,
                "a parameter cannot be `void`",
            ));
        }
        fn ownership(ty: &ast::TypeExpr) -> Ownership {
            match &ty.kind {
                TypeKind::Tuple(parts) => Ownership::Tuple(parts.iter().map(ownership).collect()),
                TypeKind::Shaped {
                    head: ShapedHead::Tensor,
                    ..
                } => Ownership::Owned,
                TypeKind::Shaped {
                    head: ShapedHead::SharedTensor,
                    ..
                } => Ownership::Shared,
                TypeKind::Shaped {
                    head: ShapedHead::MutTensor,
                    ..
                } => Ownership::Exclusive,
                _ => Ownership::Value,
            }
        }
        let ownership = ownership(&p.ty);
        if ownership == Ownership::Exclusive {
            if let ValueType::Tensor(tensor) = &ty {
                if let Elem::Repr(representation) = &tensor.elem {
                    if crate::registry::representation_info(*representation).access
                        != crate::registry::RepresentationAccess::ReadWrite
                    {
                        return Err(Diagnostic::with_rule(
DiagnosticRule::Type,
                            p.ty.span,
                            format!(
                                "representation `{}` is decode-only and cannot be a mutable tensor parameter",
                                crate::registry::representation_info(*representation).name
                            ),
                        ));
                    }
                }
            }
        }
        params.push(SigParam {
            name: p.name.name.clone(),
            ownership,
            ty,
            span: p.name.span,
        });
    }
    fn borrowed_result(ty: &ast::TypeExpr) -> bool {
        match &ty.kind {
            TypeKind::Shaped {
                head: ShapedHead::SharedTensor | ShapedHead::MutTensor,
                ..
            } => true,
            TypeKind::Tuple(parts) => parts.iter().any(borrowed_result),
            _ => false,
        }
    }
    let result = match &s.result {
        Some(t) if borrowed_result(t) => {
            return Err(Diagnostic::with_rule(
                DiagnosticRule::Ownership,
                t.span,
                "borrowed tensors cannot be returned; return an owned `tensor`",
            ));
        }
        Some(t) => {
            let ty = type_from_ast(
                t,
                &shape_params,
                &shape_symbols,
                &mut arena,
                &mut elem_params,
            )?;
            reject_rank_zero_packed(&ty, t.span)?;
            ty
        }
        None => ValueType::Void,
    };
    collect_body_element_parameters(body, &mut elem_params);
    let mut predicates = Vec::new();
    for e in s.predicates.iter().chain(extra_predicates) {
        predicates_of(
            e,
            &shape_params,
            &shape_symbols,
            &mut arena,
            &mut predicates,
        )?;
    }
    let dimensions = shape_params
        .iter()
        .zip(&shape_symbols)
        .map(|(name, &symbol)| {
            let value = arena.int_symbol(symbol);
            let admits_zero = predicates.iter().any(|conjunct: &WhereConjunct| {
                matches!(conjunct.predicate, Predicate::NonNegative(expression) if super::prove::same(&arena, expression, value))
            });
            SignatureDimension {
                name: name.clone(),
                symbol,
                admits_zero,
            }
        })
        .collect();
    Ok(Sig {
        name: name.to_string(),
        dimensions,
        elem_params,
        params,
        result,
        predicates,
        arena,
    })
}

/// Element parameters are implicit lexical binders wherever a tensor element
/// descriptor occurs, including owned allocations in the body. Discover them
/// during resolution so call contracts and generated entry bindings are
/// complete before body checking begins.
fn collect_body_element_parameters(block: &ast::Block, out: &mut Vec<String>) {
    fn element(name: &ast::Ident, out: &mut Vec<String>) {
        if DType::from_name(&name.name).is_none()
            && registry::representation(&name.name).is_none()
            && name
                .name
                .chars()
                .next()
                .is_some_and(|first| first.is_ascii_uppercase())
            && !out.contains(&name.name)
        {
            out.push(name.name.clone());
        }
    }

    fn index(index: &ast::Index, out: &mut Vec<String>) {
        match index {
            ast::Index::Expr(expression) => expr(expression, out),
            ast::Index::Slice { start, end } => {
                if let Some(start) = start {
                    expr(start, out);
                }
                if let Some(end) = end {
                    expr(end, out);
                }
            }
        }
    }

    fn expr(expression: &ast::Expr, out: &mut Vec<String>) {
        match &expression.kind {
            ast::ExprKind::Tuple(items) => {
                for item in items {
                    expr(item, out);
                }
            }
            ast::ExprKind::Range { lo, hi } => {
                expr(lo, out);
                expr(hi, out);
            }
            ast::ExprKind::Tensor { shape, elem } => {
                for dimension in shape {
                    expr(dimension, out);
                }
                element(elem, out);
            }
            ast::ExprKind::Call {
                callee,
                bindings,
                args,
            } => {
                expr(callee, out);
                for (_, value) in bindings {
                    expr(value, out);
                }
                for argument in args {
                    expr(&argument.value, out);
                }
            }
            ast::ExprKind::Index { base, indices } => {
                expr(base, out);
                for item in indices {
                    index(item, out);
                }
            }
            ast::ExprKind::Attr { base, .. } => expr(base, out),
            ast::ExprKind::Unary { expr: operand, .. } => expr(operand, out),
            ast::ExprKind::Binary { lhs, rhs, .. } => {
                expr(lhs, out);
                expr(rhs, out);
            }
            ast::ExprKind::Int(_)
            | ast::ExprKind::Float(_)
            | ast::ExprKind::Inf
            | ast::ExprKind::Bool(_)
            | ast::ExprKind::Name(_) => {}
        }
    }

    for statement in &block.stmts {
        match &statement.kind {
            ast::StmtKind::Let { value, .. } => expr(value, out),
            ast::StmtKind::Assign { target, value, .. } => {
                expr(target, out);
                expr(value, out);
            }
            ast::StmtKind::For { iter, body, .. } => {
                expr(iter, out);
                collect_body_element_parameters(body, out);
            }
            ast::StmtKind::If { cond, then, els } => {
                expr(cond, out);
                collect_body_element_parameters(then, out);
                if let Some(els) = els {
                    collect_body_element_parameters(els, out);
                }
            }
            ast::StmtKind::Return(values) => {
                for value in values {
                    expr(value, out);
                }
            }
            ast::StmtKind::Expr(expression) => expr(expression, out),
        }
    }
}

fn reject_rank_zero_packed(ty: &ValueType, span: Span) -> Result<(), Diagnostic> {
    match ty {
        ValueType::Tensor(tensor)
            if tensor.axes.is_empty() && matches!(tensor.elem, Elem::Repr(_)) =>
        {
            Err(Diagnostic::with_rule(
DiagnosticRule::Type,
                span,
                "a packed tensor must have at least one axis because packets run along the last axis",
            ))
        }
        ValueType::Tuple(items) => {
            for item in items.iter() {
                reject_rank_zero_packed(item, span)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn elems_overlap(a: &Elem, b: &Elem) -> bool {
    matches!((a, b), (Elem::Param(_), _) | (_, Elem::Param(_))) || a == b
}

fn constant(arena: &ExprArena, value: IntExpr) -> Option<i64> {
    super::prove::constant(arena, value)
}

fn kinds_overlap(a: &ValueType, aa: &ExprArena, b: &ValueType, ba: &ExprArena) -> bool {
    match (a, b) {
        (
            ValueType::Scalar(_) | ValueType::Index { .. },
            ValueType::Scalar(_) | ValueType::Index { .. },
        ) => a.scalar_dtype() == b.scalar_dtype(),
        (ValueType::Range { .. }, ValueType::Range { .. }) => true,
        (ValueType::Tensor(x), ValueType::Tensor(y)) => {
            x.rank() == y.rank()
                && elems_overlap(&x.elem, &y.elem)
                && x.axes.iter().zip(&y.axes).all(|(p, q)| {
                    match (constant(aa, *p), constant(ba, *q)) {
                        (Some(m), Some(n)) => m == n,
                        _ => true,
                    }
                })
        }
        (ValueType::Tuple(x), ValueType::Tuple(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(p, q)| kinds_overlap(p, aa, q, ba))
        }
        (
            ValueType::Opaque {
                capability: ac,
                name: an,
            },
            ValueType::Opaque {
                capability: bc,
                name: bn,
            },
        ) => ac == bc && an == bn,
        (ValueType::Void, ValueType::Void) => true,
        _ => false,
    }
}

pub(crate) fn structures_overlap(a: &Sig, b: &Sig) -> bool {
    a.params.len() == b.params.len()
        && a.params.iter().zip(&b.params).all(|(p, q)| {
            p.ownership == q.ownership && kinds_overlap(&p.ty, &a.arena, &q.ty, &b.arena)
        })
}

/// Overlapping definitions agree on result, ownership and exact symbolic
/// shape/element relationships after ordinal dimension renaming.
fn contract_mismatch(a: &Sig, b: &Sig) -> Option<String> {
    if let Some((p, q)) = a
        .params
        .iter()
        .zip(&b.params)
        .find(|(p, q)| p.ownership != q.ownership)
    {
        return Some(format!(
            "parameter `{}` has different ownership than `{}` of an overlapping definition of `{}`",
            q.name, p.name, a.name
        ));
    }
    if let Some((p, q)) = a
        .params
        .iter()
        .zip(&b.params)
        .find(|(p, q)| p.name != q.name)
    {
        return Some(format!(
            "parameter names differ from the contract of `{}`: `{}` here, `{}` in the contract",
            a.name, q.name, p.name
        ));
    }
    if a.dimensions.len() != b.dimensions.len() {
        return Some(format!(
            "overlapping definitions of `{}` must declare the same number of shape parameters",
            a.name
        ));
    }

    let mut common = ExprArena::new();
    let common_symbols: Vec<_> = a
        .dimensions
        .iter()
        .enumerate()
        .map(|(ordinal, _)| {
            common
                .template_dimension(
                    u32::try_from(ordinal).expect("signature has more than u32::MAX dimensions"),
                )
                .0
        })
        .collect();
    let mut map_a = |symbol: SymbolId, arena: &mut ExprArena| {
        let ordinal = a
            .dimensions
            .iter()
            .position(|dimension| dimension.symbol == symbol)
            .unwrap_or_else(|| panic!("signature expression mentions a non-dimension symbol"));
        AnyExpr::Int(arena.int_symbol(common_symbols[ordinal]))
    };
    let mut map_b = |symbol: SymbolId, arena: &mut ExprArena| {
        let ordinal = b
            .dimensions
            .iter()
            .position(|dimension| dimension.symbol == symbol)
            .unwrap_or_else(|| panic!("signature expression mentions a non-dimension symbol"));
        AnyExpr::Int(arena.int_symbol(common_symbols[ordinal]))
    };

    #[derive(Default)]
    struct Elements {
        bindings: HashMap<String, Elem>,
    }
    impl Elements {
        fn constrain(&mut self, contract: &Elem, implementation: &Elem) -> bool {
            match contract {
                Elem::Param(name) => match self.bindings.get(name) {
                    Some(bound) => bound == implementation,
                    None => {
                        self.bindings.insert(name.clone(), implementation.clone());
                        true
                    }
                },
                concrete => concrete == implementation,
            }
        }
    }

    fn equivalent_type(
        a: &ValueType,
        aa: &ExprArena,
        map_a: &mut super::xfer::SymbolMap<'_>,
        b: &ValueType,
        ba: &ExprArena,
        map_b: &mut super::xfer::SymbolMap<'_>,
        common: &mut ExprArena,
        elements: &mut Elements,
    ) -> bool {
        let mut equal = |left: IntExpr, right: IntExpr, common: &mut ExprArena| {
            let left = super::xfer::transfer_int(aa, left, common, map_a);
            let right = super::xfer::transfer_int(ba, right, common, map_b);
            left == right
        };
        match (a, b) {
            (ValueType::Scalar(x), ValueType::Scalar(y)) => x == y,
            (ValueType::Index { bound: x }, ValueType::Index { bound: y })
            | (ValueType::Range { bound: x }, ValueType::Range { bound: y }) => {
                equal(*x, *y, common)
            }
            (ValueType::Tensor(x), ValueType::Tensor(y)) => {
                x.axes.len() == y.axes.len()
                    && x.axes
                        .iter()
                        .zip(&y.axes)
                        .all(|(l, r)| equal(*l, *r, common))
                    && elements.constrain(&x.elem, &y.elem)
            }
            (ValueType::Tuple(x), ValueType::Tuple(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|(l, r)| equivalent_type(l, aa, map_a, r, ba, map_b, common, elements))
            }
            (
                ValueType::Opaque {
                    capability: ac,
                    name: an,
                },
                ValueType::Opaque {
                    capability: bc,
                    name: bn,
                },
            ) => ac == bc && an == bn,
            (ValueType::Void, ValueType::Void) => true,
            _ => false,
        }
    }

    let mut elements = Elements::default();
    if !a.params.iter().zip(&b.params).all(|(x, y)| {
        equivalent_type(
            &x.ty,
            &a.arena,
            &mut map_a,
            &y.ty,
            &b.arena,
            &mut map_b,
            &mut common,
            &mut elements,
        )
    }) {
        return Some(format!(
            "overlapping definitions of `{}` must have equivalent parameter shape and element relationships",
            a.name
        ));
    }
    if !equivalent_type(
        &a.result,
        &a.arena,
        &mut map_a,
        &b.result,
        &b.arena,
        &mut map_b,
        &mut common,
        &mut elements,
    ) {
        return Some(format!(
            "overlapping definitions of `{}` must have equivalent results: {} vs {}",
            a.name, a.result, b.result
        ));
    }
    None
}

/// Concrete elements a lowering fixes where the matched contract has an element parameter.
fn elem_bindings(contract: &Sig, lowering: &Sig) -> Vec<(String, Elem)> {
    let mut out: Vec<(String, Elem)> = Vec::new();
    for (c, l) in contract.params.iter().zip(&lowering.params) {
        if let (Some(cs), Some(ls)) = (c.ty.shaped(), l.ty.shaped()) {
            if let (Elem::Param(p), concrete @ (Elem::Dtype(_) | Elem::Repr(_))) =
                (&cs.elem, &ls.elem)
            {
                if !out.iter().any(|(n, _)| n == p) {
                    out.push((p.clone(), concrete.clone()));
                }
            }
        }
    }
    out
}

pub(crate) fn resolve<'a>(
    files: &'a [(usize, ast::File)],
    program: ProgramId,
    diagnostics: &mut Vec<Located>,
) -> Resolved<'a> {
    let mut declared: Vec<Declared<'a>> = Vec::new();
    for (file, parsed) in files {
        for decl in &parsed.decls {
            let ast::Decl::Fn(f) = decl else { continue };
            if is_builtin_name(&f.name.name) {
                diagnostics.push(Located {
                    file: *file,
                    diagnostic: Diagnostic::with_rule(
                        DiagnosticRule::Resolution,
                        f.name.span,
                        format!(
                            "`{}` names a builtin operation; a definition cannot take a builtin's name",
                            f.name.name
                        ),
                    ),
                });
                continue;
            }
            match signature_of(&f.name.name, &f.signature, &[], &f.body) {
                Ok(sig) => {
                    let requires = match requirements(&f.requires, None) {
                        Ok(requires) => requires,
                        Err(found) => {
                            diagnostics.extend(found.into_iter().map(|diagnostic| Located {
                                file: *file,
                                diagnostic,
                            }));
                            continue;
                        }
                    };
                    declared.push(Declared {
                        sig,
                        kind: DefKind::Body,
                        requires,
                        family: FamilyId::new(program, 0),
                        elem_bindings: Vec::new(),
                        body: &f.body,
                        file: *file,
                        span: f.span,
                        name_span: f.name.span,
                    })
                }
                Err(diagnostic) => diagnostics.push(Located {
                    file: *file,
                    diagnostic,
                }),
            }
        }
    }

    // Contract families: connected components of same-name overloads with overlapping structure.
    let mut component: Vec<usize> = (0..declared.len()).collect();
    fn root(component: &mut [usize], mut i: usize) -> usize {
        while component[i] != i {
            component[i] = component[component[i]];
            i = component[i];
        }
        i
    }
    for i in 0..declared.len() {
        for j in 0..i {
            if declared[i].sig.name != declared[j].sig.name
                || !structures_overlap(&declared[i].sig, &declared[j].sig)
            {
                continue;
            }
            if let Some(message) = contract_mismatch(&declared[j].sig, &declared[i].sig) {
                diagnostics.push(Located {
                    file: declared[i].file,
                    diagnostic: Diagnostic::with_rule(
                        DiagnosticRule::CallContract,
                        declared[i].name_span,
                        message,
                    ),
                });
            }
            let (a, b) = (root(&mut component, i), root(&mut component, j));
            component[a.max(b)] = a.min(b);
        }
    }
    let mut families: Vec<DeclaredFamily> = Vec::new();
    let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
    let mut family_of_root: HashMap<usize, usize> = HashMap::new();
    for i in 0..declared.len() {
        let r = root(&mut component, i);
        let family = *family_of_root.entry(r).or_insert_with(|| {
            families.push(DeclaredFamily {
                name: declared[i].sig.name.clone(),
                contract: FunctionId::new(
                    program,
                    u32::try_from(i).expect("module has more than u32::MAX definitions"),
                ),
                bodies: Vec::new(),
                lowerings: Vec::new(),
            });
            by_name
                .entry(declared[i].sig.name.clone())
                .or_default()
                .push(families.len() - 1);
            families.len() - 1
        });
        declared[i].family = FamilyId::new(
            program,
            u32::try_from(family).expect("module has more than u32::MAX families"),
        );
        let id = FunctionId::new(
            program,
            u32::try_from(i).expect("module has more than u32::MAX definitions"),
        );
        families[family].bodies.push(id);
    }

    // Lowerings attach to the portable family their restated signature overlaps.
    for (file, parsed) in files {
        for decl in &parsed.decls {
            let ast::Decl::Lower(l) = decl else { continue };
            let Some(named) = by_name.get(&l.name.name).cloned() else {
                diagnostics.push(Located { file: *file, diagnostic: Diagnostic::with_rule(DiagnosticRule::Resolution, l.name.span, format!("`{}` is not declared; a lowering implements a declared function contract", l.name.name)) });
                continue;
            };
            let target = l.target.name.clone();
            let Some(target) = BackendName::parse(&target) else {
                diagnostics.push(Located {
                    file: *file,
                    diagnostic: Diagnostic::with_rule(
                        DiagnosticRule::Resolution,
                        l.target.span,
                        format!("`{target}` is not a known target"),
                    ),
                });
                continue;
            };
            let kind = DefKind::Lower { target };
            let requires = match requirements(&l.requires, Some(&l.target.name)) {
                Ok(requires) => requires,
                Err(found) => {
                    diagnostics.extend(found.into_iter().map(|diagnostic| Located {
                        file: *file,
                        diagnostic,
                    }));
                    continue;
                }
            };
            let body = &l.body;
            let mut attach: Vec<(usize, Sig, Vec<(String, Elem)>)> = Vec::new();
            {
                let sig = match signature_of(&l.name.name, &l.signature, &l.predicates, &l.body) {
                    Ok(sig) => sig,
                    Err(diagnostic) => {
                        diagnostics.push(Located {
                            file: *file,
                            diagnostic,
                        });
                        continue;
                    }
                };
                let matching: Vec<(usize, usize)> = named
                    .iter()
                    .filter_map(|family| {
                        families[*family]
                            .bodies
                            .iter()
                            .map(|id| id.index())
                            .find(|member| {
                                matches!(declared[*member].kind, DefKind::Body)
                                    && structures_overlap(&declared[*member].sig, &sig)
                            })
                            .map(|member| (*family, member))
                    })
                    .collect();
                match matching.as_slice() {
                        [] => diagnostics.push(Located { file: *file, diagnostic: Diagnostic::with_rule(DiagnosticRule::Resolution, l.name.span, format!("no definition of `{}` has this parameter structure (kinds, ranks, element types); a lowering restates the contract it implements", l.name.name)) }),
                        [(family, member)] => {
                            let contract = &declared[*member];
                            if let Some(message) = contract_mismatch(&contract.sig, &sig) {
                                diagnostics.push(Located { file: *file, diagnostic: Diagnostic::with_rule(DiagnosticRule::CallContract, l.name.span, message) });
                            }
                            let bindings = elem_bindings(&contract.sig, &sig);
                            attach.push((*family, sig, bindings));
                        }
                        _ => diagnostics.push(Located { file: *file, diagnostic: Diagnostic::with_rule(DiagnosticRule::Resolution, l.name.span, format!("this lowering overlaps several disjoint contract families of `{}`; restate one family's parameter structure", l.name.name)) }),
                    }
            }
            for (family, sig, elem_bindings) in attach {
                families[family].lowerings.push(FunctionId::new(
                    program,
                    u32::try_from(declared.len())
                        .expect("module has more than u32::MAX definitions"),
                ));
                declared.push(Declared {
                    sig,
                    kind: kind.clone(),
                    requires: requires.clone(),
                    family: FamilyId::new(
                        program,
                        u32::try_from(family).expect("module has more than u32::MAX families"),
                    ),
                    elem_bindings,
                    body,
                    file: *file,
                    span: l.span,
                    name_span: l.name.span,
                });
            }
        }
    }
    let call_graph = NameCallGraph::new(&declared, &families, &by_name);
    Resolved {
        declared,
        families,
        by_name,
        call_graph,
    }
}
