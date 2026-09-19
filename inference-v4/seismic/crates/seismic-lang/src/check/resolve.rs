//! Declarations: signatures, `where` predicates, contract families, lowering attachment
//! and explicit target coverage. One flat global namespace.

use crate::intrinsics::{self, IntrinsicResult, Semantics};
use crate::repr;
use crate::sir::{ContractFamily, DefId, DefKind, Predicate};
use crate::span::{Diagnostic, Span};
use crate::sym::{Atom, Sym};
use crate::syntax::ast::{self, BinaryOp, ExprKind as A, Mode, ShapedHead, TypeKind};
use crate::types::{DType, Elem, Extent, NativeTy, Shaped, Ty};
use std::collections::HashMap;

/// A diagnostic attributed to a source file (index into the compiled file list).
#[derive(Clone, Debug)]
pub(crate) struct Located {
    pub file: usize,
    pub diagnostic: Diagnostic,
}

#[derive(Clone, Debug)]
pub(crate) struct SigParam {
    pub name: String,
    pub mode: Mode,
    pub ty: Ty,
    pub span: Span,
}

/// A declaration's checked interface. Shapes are semantic extents over its own shape parameters.
#[derive(Clone, Debug)]
pub(crate) struct Sig {
    pub name: String,
    pub shape_params: Vec<String>,
    pub elem_params: Vec<String>,
    pub params: Vec<SigParam>,
    pub aliases: Vec<(usize, usize)>,
    pub result: Ty,
    pub predicates: Vec<Predicate>,
}

/// One definition before its body is checked.
pub(crate) struct Declared<'a> {
    pub sig: Sig,
    pub kind: DefKind,
    pub family: usize,
    pub elem_bindings: Vec<(String, Elem)>,
    pub body: &'a ast::Block,
    pub file: usize,
    pub span: Span,
    pub name_span: Span,
}

pub(crate) struct Resolved<'a> {
    pub declared: Vec<Declared<'a>>,
    pub families: Vec<ContractFamily>,
    /// Function name -> indices into `families`.
    pub by_name: HashMap<String, Vec<usize>>,
}

fn native_targets(ty: &Ty, out: &mut Vec<String>) {
    match ty {
        Ty::Native(native) => out.push(native.target.clone()),
        Ty::Tuple(items) => items.iter().for_each(|item| native_targets(item, out)),
        _ => {}
    }
}

fn check_signature_target(sig: &Sig, target: Option<&str>, span: Span) -> Result<(), Diagnostic> {
    let mut native = Vec::new();
    for param in &sig.params {
        native_targets(&param.ty, &mut native);
    }
    native_targets(&sig.result, &mut native);
    if let Some(found) = native
        .into_iter()
        .find(|found| Some(found.as_str()) != target)
    {
        return Err(Diagnostic::new(span, match target {
            Some(expected) => format!("native type for backend `{found}` cannot appear in a declaration for backend `{expected}`"),
            None => format!("native type for backend `{found}` cannot appear in a portable function signature"),
        }));
    }
    Ok(())
}

/// A shape expression: integers, shape parameters, and `+ - * / %` over them.
pub(crate) fn shape_sym(e: &ast::Expr, shape_params: &[String]) -> Result<Sym, Diagnostic> {
    match &e.kind {
        A::Int(v) => i64::try_from(*v).map(Sym::constant).map_err(|_| {
            Diagnostic::new(
                e.span,
                "shape constant does not fit a signed 64-bit integer",
            )
        }),
        A::Name(n) if shape_params.contains(&n.name) => Ok(Sym::param(&n.name)),
        A::Name(n) => Err(Diagnostic::new(
            n.span,
            format!("`{}` is not a declared shape parameter", n.name),
        )),
        A::Binary { op, lhs, rhs } => {
            let l = shape_sym(lhs, shape_params)?;
            let r = shape_sym(rhs, shape_params)?;
            match op {
                BinaryOp::Add => Ok(l.add(&r)),
                BinaryOp::Sub => Ok(l.sub(&r)),
                BinaryOp::Mul => Ok(l.mul(&r)),
                BinaryOp::Div | BinaryOp::Rem => {
                    if r.as_constant().is_some_and(|c| c <= 0) {
                        return Err(Diagnostic::new(rhs.span, "shape divisor must be positive"));
                    }
                    Ok(if *op == BinaryOp::Div {
                        l.quot(&r)
                    } else {
                        l.rem(&r)
                    })
                }
                _ => Err(Diagnostic::new(
                    e.span,
                    "only + - * / % are allowed in shapes",
                )),
            }
        }
        _ => Err(Diagnostic::new(
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
    if repr::lookup(&name.name).is_some() {
        return Ok(Elem::Repr(name.name.clone()));
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
    Err(Diagnostic::new(
        name.span,
        format!(
            "`{}` is not a dtype, a representation, or an element parameter",
            name.name
        ),
    ))
}

/// The native type `target.name(args)` from the target's intrinsic table.
pub(crate) fn native_type(
    target: &ast::Ident,
    name: &ast::Ident,
    args: &[ast::Expr],
    span: Span,
) -> Result<NativeTy, Diagnostic> {
    let Some(table) = intrinsics::table(&target.name) else {
        return Err(Diagnostic::new(
            target.span,
            format!("`{}` is not a target namespace", target.name),
        ));
    };
    let Some(intrinsic) = table.iter().find(|i| {
        i.operation.name() == name.name && matches!(i.result, IntrinsicResult::Frag8x8OfNamedDtype)
    }) else {
        return Err(Diagnostic::new(
            name.span,
            format!(
                "`{}.{}` is not a native type of target `{}`",
                target.name, name.name, target.name
            ),
        ));
    };
    let [arg] = args else {
        return Err(Diagnostic::new(
            span,
            format!("`{}.{}` takes one dtype name", target.name, name.name),
        ));
    };
    let dtype = match &arg.kind {
        A::Name(n) => DType::from_name(&n.name),
        _ => None,
    };
    let Some(dtype) = dtype else {
        return Err(Diagnostic::new(arg.span, "expected a dtype name"));
    };
    let Semantics::Fragment { rows, columns } = intrinsic.operation.semantics() else {
        return Err(Diagnostic::new(
            name.span,
            format!(
                "`{}.{}` does not declare a native fragment",
                target.name, name.name
            ),
        ));
    };
    Ok(NativeTy {
        target: target.name.clone(),
        name: name.name.clone(),
        shape: vec![Sym::constant(rows as i64), Sym::constant(columns as i64)],
        elem: Some(Elem::Dtype(dtype)),
    })
}

fn type_from_ast(
    t: &ast::TypeExpr,
    shape_params: &[String],
    elem_params: &mut Vec<String>,
) -> Result<Ty, Diagnostic> {
    match &t.kind {
        TypeKind::Scalar(name) => match DType::from_name(&name.name) {
            Some(d) => Ok(Ty::Scalar(d)),
            None if name.name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) => Err(Diagnostic::new(name.span, format!("element parameter `{}` cannot be a scalar type: scalar values have a concrete dtype", name.name))),
            None => Err(Diagnostic::new(name.span, format!("unknown type `{}`", name.name))),
        },
        TypeKind::Index(bound) => Ok(Ty::Index(shape_sym(bound, shape_params)?)),
        TypeKind::Shaped { head, shape, elem } => {
            if shape.is_empty() {
                return Err(Diagnostic::new(t.span, "a tensor, view or tile type needs a shape"));
            }
            let mut axes = Vec::new();
            for e in shape {
                axes.push(Extent::Semantic(shape_sym(e, shape_params)?));
            }
            let shaped = Shaped::new(axes, elem_of(elem, elem_params)?);
            Ok(match head {
                ShapedHead::Tensor => Ty::Tensor(shaped),
                ShapedHead::View => Ty::View(shaped),
                ShapedHead::Tile => Ty::Tile(shaped),
            })
        }
        TypeKind::Tuple(items) => {
            let mut out = Vec::new();
            for item in items {
                let ty = type_from_ast(item, shape_params, elem_params)?;
                if ty == Ty::Void {
                    return Err(Diagnostic::new(item.span, "`void` is not a tuple component"));
                }
                out.push(ty);
            }
            Ok(Ty::Tuple(out))
        }
        TypeKind::Void => Ok(Ty::Void),
        TypeKind::Native { target, name, args } => Ok(Ty::Native(native_type(target, name, args, t.span)?)),
    }
}

/// Conjuncts of a `where` clause: comparisons, divisibility and equalities over shape
/// parameters and integer literals, plus `full(X)`.
fn predicates_of(
    e: &ast::Expr,
    shape_params: &[String],
    out: &mut Vec<Predicate>,
) -> Result<(), Diagnostic> {
    match &e.kind {
        A::Binary { op: BinaryOp::And, lhs, rhs } => {
            predicates_of(lhs, shape_params, out)?;
            predicates_of(rhs, shape_params, out)
        }
        A::Binary { op, lhs, rhs } => {
            let l = shape_sym(lhs, shape_params)?;
            let r = shape_sym(rhs, shape_params)?;
            let one = Sym::constant(1);
            out.push(match op {
                BinaryOp::Ge => Predicate::NonNegative(l.sub(&r)),
                BinaryOp::Gt => Predicate::NonNegative(l.sub(&r).sub(&one)),
                BinaryOp::Le => Predicate::NonNegative(r.sub(&l)),
                BinaryOp::Lt => Predicate::NonNegative(r.sub(&l).sub(&one)),
                BinaryOp::Eq => Predicate::Zero(l.sub(&r)),
                BinaryOp::Ne => Predicate::NonZero(l.sub(&r)),
                _ => return Err(Diagnostic::new(e.span, "a `where` predicate is a conjunction of comparisons, divisibility and equalities over shape parameters, or `full(X)`")),
            });
            Ok(())
        }
        A::Call { callee, bindings, args } if matches!(&callee.kind, A::Name(n) if n.name == "full") => {
            let param = match (bindings.is_empty(), args.as_slice()) {
                (true, [ast::Arg { name: None, value }]) => match &value.kind {
                    A::Name(n) if shape_params.contains(&n.name) => Some(n.name.clone()),
                    _ => None,
                },
                _ => None,
            };
            match param {
                Some(p) => {
                    out.push(Predicate::Full(p));
                    Ok(())
                }
                None => Err(Diagnostic::new(e.span, "`full(X)` takes one shape parameter")),
            }
        }
        _ => Err(Diagnostic::new(e.span, "a `where` predicate is a conjunction of comparisons, divisibility and equalities over shape parameters, or `full(X)`")),
    }
}

pub(crate) fn signature_of(
    name: &str,
    s: &ast::Signature,
    extra_predicates: &[ast::Expr],
) -> Result<Sig, Diagnostic> {
    let mut shape_params: Vec<String> = Vec::new();
    for p in &s.shape {
        if shape_params.contains(&p.name) {
            return Err(Diagnostic::new(
                p.span,
                format!("duplicate shape parameter `{}`", p.name),
            ));
        }
        shape_params.push(p.name.clone());
    }
    let mut elem_params = Vec::new();
    let mut params: Vec<SigParam> = Vec::new();
    for p in &s.params {
        if params.iter().any(|q| q.name == p.name.name) || shape_params.contains(&p.name.name) {
            return Err(Diagnostic::new(
                p.name.span,
                format!("duplicate parameter `{}`", p.name.name),
            ));
        }
        let ty = type_from_ast(&p.ty, &shape_params, &mut elem_params)?;
        if ty == Ty::Void {
            return Err(Diagnostic::new(p.ty.span, "a parameter cannot be `void`"));
        }
        if p.mode != Mode::In
            && !matches!(
                ty,
                Ty::Tensor(_) | Ty::View(_) | Ty::Tile(_) | Ty::Native(_)
            )
        {
            return Err(Diagnostic::new(
                p.ty.span,
                format!(
                    "`out`/`inout` applies to tensors, views, tiles and native values, not {ty}"
                ),
            ));
        }
        params.push(SigParam {
            name: p.name.name.clone(),
            mode: p.mode,
            ty,
            span: p.name.span,
        });
    }
    let mut aliases = Vec::new();
    for (a, b) in &s.aliases {
        let find = |id: &ast::Ident| {
            params
                .iter()
                .position(|p| p.name == id.name)
                .ok_or_else(|| {
                    Diagnostic::new(
                        id.span,
                        format!("`alias` names unknown parameter `{}`", id.name),
                    )
                })
        };
        aliases.push((find(a)?, find(b)?));
    }
    let result = match &s.result {
        Some(t) => type_from_ast(t, &shape_params, &mut elem_params)?,
        None => Ty::Void,
    };
    let mut predicates = Vec::new();
    for e in s.predicates.iter().chain(extra_predicates) {
        predicates_of(e, &shape_params, &mut predicates)?;
    }
    Ok(Sig {
        name: name.to_string(),
        shape_params,
        elem_params,
        params,
        aliases,
        result,
        predicates,
    })
}

fn elems_overlap(a: &Elem, b: &Elem) -> bool {
    matches!((a, b), (Elem::Param(_), _) | (_, Elem::Param(_))) || a == b
}

/// Whether two types can describe the same argument: kinds, ranks, element descriptors and
/// constant extents. Shape relationships between parameters are not compared.
pub(crate) fn kinds_overlap(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Scalar(_) | Ty::Index(_), Ty::Scalar(_) | Ty::Index(_)) => {
            a.scalar_dtype() == b.scalar_dtype()
        }
        (Ty::Tensor(x), Ty::Tensor(y))
        | (Ty::View(x), Ty::View(y))
        | (Ty::Tile(x), Ty::Tile(y)) => {
            x.rank() == y.rank()
                && elems_overlap(&x.elem, &y.elem)
                && x.axes.iter().zip(&y.axes).all(|(p, q)| {
                    match (
                        p.semantic().and_then(Sym::as_constant),
                        q.semantic().and_then(Sym::as_constant),
                    ) {
                        (Some(m), Some(n)) => m == n,
                        _ => true,
                    }
                })
        }
        (Ty::Tuple(x), Ty::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| kinds_overlap(p, q))
        }
        (Ty::Native(x), Ty::Native(y)) => x == y,
        (Ty::Void, Ty::Void) => true,
        _ => false,
    }
}

pub(crate) fn structures_overlap(a: &Sig, b: &Sig) -> bool {
    a.params.len() == b.params.len()
        && a.params
            .iter()
            .zip(&b.params)
            .all(|(p, q)| kinds_overlap(&p.ty, &q.ty))
}

/// Overlapping definitions agree on result, modes and shape/element relationships.
fn contract_mismatch(a: &Sig, b: &Sig) -> Option<String> {
    if let Some((p, q)) = a
        .params
        .iter()
        .zip(&b.params)
        .find(|(p, q)| p.mode != q.mode)
    {
        return Some(format!(
            "parameter `{}` has a different mode than `{}` of an overlapping definition of `{}`",
            q.name, p.name, a.name
        ));
    }
    let normalize_aliases = |aliases: &[(usize, usize)]| {
        let mut normalized: Vec<(usize, usize)> = aliases
            .iter()
            .map(|&(left, right)| (left.min(right), left.max(right)))
            .collect();
        normalized.sort_unstable();
        normalized.dedup();
        normalized
    };
    if normalize_aliases(&a.aliases) != normalize_aliases(&b.aliases) {
        return Some(format!(
            "overlapping definitions of `{}` must declare identical `alias` permissions",
            a.name
        ));
    }
    if a.shape_params.len() != b.shape_params.len() {
        return Some(format!(
            "overlapping definitions of `{}` must declare the same number of shape parameters",
            a.name
        ));
    }

    fn rename_shape(sym: &Sym, names: &HashMap<String, String>) -> Sym {
        let mut out = Sym::constant(0);
        for (monomial, coefficient) in sym.monomials() {
            let mut term = Sym::constant(coefficient);
            for (atom, power) in monomial {
                let renamed = match atom {
                    Atom::Param(name) => Sym::param(names.get(name).map_or(name, String::as_str)),
                    Atom::Quot(numerator, denominator) => {
                        rename_shape(numerator, names).quot(&rename_shape(denominator, names))
                    }
                    Atom::Rem(numerator, denominator) => {
                        rename_shape(numerator, names).rem(&rename_shape(denominator, names))
                    }
                };
                for _ in 0..*power {
                    term = term.mul(&renamed);
                }
            }
            out = out.add(&term);
        }
        out
    }

    #[derive(Default)]
    struct Elements {
        implementations: HashMap<String, Elem>,
    }
    impl Elements {
        fn constrain(&mut self, contract: &Elem, implementation: &Elem) -> bool {
            match contract {
                Elem::Param(name) => match self.implementations.get(name) {
                    Some(bound) => bound == implementation,
                    None => {
                        self.implementations
                            .insert(name.clone(), implementation.clone());
                        true
                    }
                },
                concrete => concrete == implementation,
            }
        }
    }

    fn equivalent_type(
        a: &Ty,
        b: &Ty,
        shape_names: &HashMap<String, String>,
        elements: &mut Elements,
    ) -> bool {
        match (a, b) {
            (Ty::Scalar(a), Ty::Scalar(b)) => a == b,
            (Ty::Index(a), Ty::Index(b)) => a == &rename_shape(b, shape_names),
            (Ty::Tensor(a), Ty::Tensor(b))
            | (Ty::View(a), Ty::View(b))
            | (Ty::Tile(a), Ty::Tile(b)) => {
                a.axes.len() == b.axes.len()
                    && a.axes.iter().zip(&b.axes).all(|(a, b)| {
                        matches!((a.semantic(), b.semantic()), (Some(a), Some(b)) if a == &rename_shape(b, shape_names))
                    })
                    && elements.constrain(&a.elem, &b.elem)
            }
            (Ty::Tuple(a), Ty::Tuple(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b)
                        .all(|(a, b)| equivalent_type(a, b, shape_names, elements))
            }
            (Ty::Native(a), Ty::Native(b)) => a == b,
            (Ty::Void, Ty::Void) => true,
            _ => false,
        }
    }

    let shape_names: HashMap<String, String> = b
        .shape_params
        .iter()
        .cloned()
        .zip(a.shape_params.iter().cloned())
        .collect();
    let mut elements = Elements::default();
    if !a
        .params
        .iter()
        .zip(&b.params)
        .all(|(a, b)| equivalent_type(&a.ty, &b.ty, &shape_names, &mut elements))
    {
        return Some(format!(
            "overlapping definitions of `{}` must have equivalent parameter shape and element relationships",
            a.name
        ));
    }
    if !equivalent_type(&a.result, &b.result, &shape_names, &mut elements) {
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
    diagnostics: &mut Vec<Located>,
) -> Resolved<'a> {
    let mut declared: Vec<Declared<'a>> = Vec::new();
    for (file, parsed) in files {
        for decl in &parsed.decls {
            let ast::Decl::Fn(f) = decl else { continue };
            if let Some(target) = &f.target {
                if intrinsics::table(&target.name).is_none() {
                    diagnostics.push(Located {
                        file: *file,
                        diagnostic: Diagnostic::new(
                            target.span,
                            format!("`{}` is not a known target", target.name),
                        ),
                    });
                    continue;
                }
            }
            match signature_of(&f.name.name, &f.signature, &[]) {
                Ok(sig) => {
                    if let Err(diagnostic) = check_signature_target(
                        &sig,
                        f.target.as_ref().map(|t| t.name.as_str()),
                        f.name.span,
                    ) {
                        diagnostics.push(Located {
                            file: *file,
                            diagnostic,
                        });
                        continue;
                    }
                    declared.push(Declared {
                        sig,
                        kind: DefKind::Body {
                            target: f.target.as_ref().map(|t| t.name.clone()),
                        },
                        family: 0,
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
            if declared[i].kind.target().is_some() != declared[j].kind.target().is_some() {
                diagnostics.push(Located {
                    file: declared[i].file,
                    diagnostic: Diagnostic::new(
                        declared[i].name_span,
                        format!("portable and backend-specific functions named `{}` overlap; a backend-specific function is a separate helper, not an implementation of a portable family", declared[i].sig.name),
                    ),
                });
            }
            if let Some(message) = contract_mismatch(&declared[j].sig, &declared[i].sig) {
                diagnostics.push(Located {
                    file: declared[i].file,
                    diagnostic: Diagnostic::new(declared[i].name_span, message),
                });
            }
            let (a, b) = (root(&mut component, i), root(&mut component, j));
            component[a.max(b)] = a.min(b);
        }
    }
    let mut families: Vec<ContractFamily> = Vec::new();
    let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
    let mut family_of_root: HashMap<usize, usize> = HashMap::new();
    for i in 0..declared.len() {
        let r = root(&mut component, i);
        let family = *family_of_root.entry(r).or_insert_with(|| {
            families.push(ContractFamily {
                name: declared[i].sig.name.clone(),
                bodies: Vec::new(),
                lowerings: Vec::new(),
            });
            by_name
                .entry(declared[i].sig.name.clone())
                .or_default()
                .push(families.len() - 1);
            families.len() - 1
        });
        declared[i].family = family;
        let id = DefId(i as u32);
        families[family].bodies.push(id);
    }

    // Lowerings attach to the portable family their restated signature overlaps.
    for (file, parsed) in files {
        for decl in &parsed.decls {
            let ast::Decl::Lower(l) = decl else { continue };
            let Some(named) = by_name.get(&l.name.name).cloned() else {
                diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, format!("`{}` is not declared; a lowering implements a declared function contract", l.name.name)) });
                continue;
            };
            let target = l.target.name.clone();
            if intrinsics::table(&target).is_none() {
                diagnostics.push(Located {
                    file: *file,
                    diagnostic: Diagnostic::new(
                        l.target.span,
                        format!("`{target}` is not a known target"),
                    ),
                });
                continue;
            }
            let kind = DefKind::Lower { target };
            let body = &l.body;
            let mut attach: Vec<(usize, Sig, Vec<(String, Elem)>)> = Vec::new();
            {
                let sig = match signature_of(&l.name.name, &l.signature, &l.predicates) {
                    Ok(sig) => sig,
                    Err(diagnostic) => {
                        diagnostics.push(Located {
                            file: *file,
                            diagnostic,
                        });
                        continue;
                    }
                };
                if let Err(diagnostic) =
                    check_signature_target(&sig, Some(&l.target.name), l.name.span)
                {
                    diagnostics.push(Located {
                        file: *file,
                        diagnostic,
                    });
                    continue;
                }
                let matching: Vec<(usize, usize)> = named
                    .iter()
                    .filter_map(|family| {
                        families[*family]
                            .bodies
                            .iter()
                            .map(|id| id.0 as usize)
                            .find(|member| {
                                matches!(declared[*member].kind, DefKind::Body { target: None })
                                    && structures_overlap(&declared[*member].sig, &sig)
                            })
                            .map(|member| (*family, member))
                    })
                    .collect();
                match matching.as_slice() {
                        [] => diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, format!("no definition of `{}` has this parameter structure (kinds, ranks, element types); a lowering restates the contract it implements", l.name.name)) }),
                        [(family, member)] => {
                            let contract = &declared[*member];
                            if let Some(message) = contract_mismatch(&contract.sig, &sig) {
                                diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, message) });
                            }
                            let bindings = elem_bindings(&contract.sig, &sig);
                            attach.push((*family, sig, bindings));
                        }
                        _ => diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, format!("this lowering overlaps several disjoint contract families of `{}`; restate one family's parameter structure", l.name.name)) }),
                    }
            }
            for (family, sig, elem_bindings) in attach {
                families[family]
                    .lowerings
                    .push(DefId(declared.len() as u32));
                declared.push(Declared {
                    sig,
                    kind: kind.clone(),
                    family,
                    elem_bindings,
                    body,
                    file: *file,
                    span: l.span,
                    name_span: l.name.span,
                });
            }
        }
    }
    Resolved {
        declared,
        families,
        by_name,
    }
}
