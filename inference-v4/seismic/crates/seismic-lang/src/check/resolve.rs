//! Declarations: signatures, `where` predicates, contract families, lowering attachment
//! and explicit target coverage. One flat global namespace.

use crate::intrinsics::{self, IntrinsicResult, Semantics};
use crate::repr;
use crate::span::{Diagnostic, Span};
use crate::sir::{ContractFamily, DefId, DefKind, Predicate};
use crate::syntax::ast::{self, BinaryOp, ExprKind as A, Mode, ShapedHead, TypeKind};
use crate::types::{DType, Elem, Extent, NativeTy, Shaped, Ty};
use crate::sym::Sym;
use crate::Scope;
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
    pub admit: bool,
    pub export: bool,
    pub elem_bindings: Vec<(String, Elem)>,
    pub body: Option<&'a ast::Block>,
    pub file: usize,
    pub scope: Scope,
    pub span: Span,
    pub name_span: Span,
}

pub(crate) struct Resolved<'a> {
    pub declared: Vec<Declared<'a>>,
    pub families: Vec<ContractFamily>,
    /// Function name -> indices into `families`.
    pub by_name: HashMap<String, Vec<usize>>,
}

/// A shape expression: integers, shape parameters, and `+ - * / %` over them.
pub(crate) fn shape_sym(e: &ast::Expr, shape_params: &[String]) -> Result<Sym, Diagnostic> {
    match &e.kind {
        A::Int(v) => i64::try_from(*v).map(Sym::constant).map_err(|_| Diagnostic::new(e.span, "shape constant does not fit a signed 64-bit integer")),
        A::Name(n) if shape_params.contains(&n.name) => Ok(Sym::param(&n.name)),
        A::Name(n) => Err(Diagnostic::new(n.span, format!("`{}` is not a declared shape parameter", n.name))),
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
                    Ok(if *op == BinaryOp::Div { l.quot(&r) } else { l.rem(&r) })
                }
                _ => Err(Diagnostic::new(e.span, "only + - * / % are allowed in shapes")),
            }
        }
        _ => Err(Diagnostic::new(e.span, "a shape is an integer expression over shape parameters")),
    }
}

/// Element descriptor: dtype, representation, or an implicit element parameter (capitalized name).
pub(crate) fn elem_of(name: &ast::Ident, elem_params: &mut Vec<String>) -> Result<Elem, Diagnostic> {
    if let Some(d) = DType::from_name(&name.name) {
        return Ok(Elem::Dtype(d));
    }
    if repr::lookup(&name.name).is_some() {
        return Ok(Elem::Repr(name.name.clone()));
    }
    if name.name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        if !elem_params.contains(&name.name) {
            elem_params.push(name.name.clone());
        }
        return Ok(Elem::Param(name.name.clone()));
    }
    Err(Diagnostic::new(name.span, format!("`{}` is not a dtype, a representation, or an element parameter", name.name)))
}

/// The native type `target.name(args)` from the target's intrinsic table.
pub(crate) fn native_type(target: &ast::Ident, name: &ast::Ident, args: &[ast::Expr], span: Span) -> Result<NativeTy, Diagnostic> {
    let Some(table) = intrinsics::table(&target.name) else {
        return Err(Diagnostic::new(target.span, format!("`{}` is not a target namespace", target.name)));
    };
    let Some(intrinsic) = table.iter().find(|i| i.operation.name() == name.name && matches!(i.result, IntrinsicResult::Frag8x8OfNamedDtype)) else {
        return Err(Diagnostic::new(name.span, format!("`{}.{}` is not a native type of target `{}`", target.name, name.name, target.name)));
    };
    let [arg] = args else {
        return Err(Diagnostic::new(span, format!("`{}.{}` takes one dtype name", target.name, name.name)));
    };
    let dtype = match &arg.kind {
        A::Name(n) => DType::from_name(&n.name),
        _ => None,
    };
    let Some(dtype) = dtype else {
        return Err(Diagnostic::new(arg.span, "expected a dtype name"));
    };
    let Semantics::Fragment { rows, columns } = intrinsic.operation.semantics() else {
        return Err(Diagnostic::new(name.span, format!("`{}.{}` does not declare a native fragment", target.name, name.name)));
    };
    Ok(NativeTy { target: target.name.clone(), name: name.name.clone(), shape: vec![Sym::constant(rows as i64), Sym::constant(columns as i64)], elem: Some(Elem::Dtype(dtype)) })
}

fn type_from_ast(t: &ast::TypeExpr, shape_params: &[String], elem_params: &mut Vec<String>) -> Result<Ty, Diagnostic> {
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
fn predicates_of(e: &ast::Expr, shape_params: &[String], out: &mut Vec<Predicate>) -> Result<(), Diagnostic> {
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

pub(crate) fn signature_of(name: &str, s: &ast::Signature, extra_predicates: &[ast::Expr]) -> Result<Sig, Diagnostic> {
    let mut shape_params: Vec<String> = Vec::new();
    for p in &s.shape {
        if shape_params.contains(&p.name) {
            return Err(Diagnostic::new(p.span, format!("duplicate shape parameter `{}`", p.name)));
        }
        shape_params.push(p.name.clone());
    }
    let mut elem_params = Vec::new();
    let mut params: Vec<SigParam> = Vec::new();
    for p in &s.params {
        if params.iter().any(|q| q.name == p.name.name) || shape_params.contains(&p.name.name) {
            return Err(Diagnostic::new(p.name.span, format!("duplicate parameter `{}`", p.name.name)));
        }
        let ty = type_from_ast(&p.ty, &shape_params, &mut elem_params)?;
        if ty == Ty::Void {
            return Err(Diagnostic::new(p.ty.span, "a parameter cannot be `void`"));
        }
        if p.mode != Mode::In && !matches!(ty, Ty::Tensor(_) | Ty::View(_) | Ty::Tile(_) | Ty::Native(_)) {
            return Err(Diagnostic::new(p.ty.span, format!("`out`/`inout` applies to tensors, views, tiles and native values, not {ty}")));
        }
        params.push(SigParam { name: p.name.name.clone(), mode: p.mode, ty, span: p.name.span });
    }
    let mut aliases = Vec::new();
    for (a, b) in &s.aliases {
        let find = |id: &ast::Ident| params.iter().position(|p| p.name == id.name).ok_or_else(|| Diagnostic::new(id.span, format!("`alias` names unknown parameter `{}`", id.name)));
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
    Ok(Sig { name: name.to_string(), shape_params, elem_params, params, aliases, result, predicates })
}

fn elems_overlap(a: &Elem, b: &Elem) -> bool {
    matches!((a, b), (Elem::Param(_), _) | (_, Elem::Param(_))) || a == b
}

/// Whether two types can describe the same argument: kinds, ranks, element descriptors and
/// constant extents. Shape relationships between parameters are not compared.
pub(crate) fn kinds_overlap(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Scalar(_) | Ty::Index(_), Ty::Scalar(_) | Ty::Index(_)) => a.scalar_dtype() == b.scalar_dtype(),
        (Ty::Tensor(x), Ty::Tensor(y)) | (Ty::View(x), Ty::View(y)) | (Ty::Tile(x), Ty::Tile(y)) => {
            x.rank() == y.rank()
                && elems_overlap(&x.elem, &y.elem)
                && x.axes.iter().zip(&y.axes).all(|(p, q)| match (p.semantic().and_then(Sym::as_constant), q.semantic().and_then(Sym::as_constant)) {
                    (Some(m), Some(n)) => m == n,
                    _ => true,
                })
        }
        (Ty::Tuple(x), Ty::Tuple(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| kinds_overlap(p, q)),
        (Ty::Native(x), Ty::Native(y)) => x == y,
        (Ty::Void, Ty::Void) => true,
        _ => false,
    }
}

pub(crate) fn structures_overlap(a: &Sig, b: &Sig) -> bool {
    a.params.len() == b.params.len() && a.params.iter().zip(&b.params).all(|(p, q)| kinds_overlap(&p.ty, &q.ty))
}

/// Overlapping definitions agree on result, modes and numerical contract.
fn contract_mismatch(a: &Sig, a_admit: bool, b: &Sig, b_admit: Option<bool>) -> Option<String> {
    if let Some((p, q)) = a.params.iter().zip(&b.params).find(|(p, q)| p.mode != q.mode) {
        return Some(format!("parameter `{}` has a different mode than `{}` of an overlapping definition of `{}`", q.name, p.name, a.name));
    }
    if !kinds_overlap(&a.result, &b.result) {
        return Some(format!("overlapping definitions of `{}` must agree on the result: {} vs {}", a.name, a.result, b.result));
    }
    if b_admit.is_some_and(|b_admit| b_admit != a_admit) {
        return Some(format!("overlapping definitions of `{}` must agree on `admit`", a.name));
    }
    None
}

/// Concrete elements a lowering fixes where the matched contract has an element parameter.
fn elem_bindings(contract: &Sig, lowering: &Sig) -> Vec<(String, Elem)> {
    let mut out: Vec<(String, Elem)> = Vec::new();
    for (c, l) in contract.params.iter().zip(&lowering.params) {
        if let (Some(cs), Some(ls)) = (c.ty.shaped(), l.ty.shaped()) {
            if let (Elem::Param(p), concrete @ (Elem::Dtype(_) | Elem::Repr(_))) = (&cs.elem, &ls.elem) {
                if !out.iter().any(|(n, _)| n == p) {
                    out.push((p.clone(), concrete.clone()));
                }
            }
        }
    }
    out
}

pub(crate) fn resolve<'a>(files: &'a [(usize, Scope, ast::File)], diagnostics: &mut Vec<Located>) -> Resolved<'a> {
    let mut declared: Vec<Declared<'a>> = Vec::new();
    for (file, scope, parsed) in files {
        for decl in &parsed.decls {
            let ast::Decl::Fn(f) = decl else { continue };
            match signature_of(&f.name.name, &f.signature, &[]) {
                Ok(sig) => declared.push(Declared {
                    sig,
                    kind: if f.body.is_some() { DefKind::Body } else { DefKind::Contract },
                    family: 0,
                    admit: f.admit,
                    export: f.export,
                    elem_bindings: Vec::new(),
                    body: f.body.as_ref(),
                    file: *file,
                    scope: scope.clone(),
                    span: f.span,
                    name_span: f.name.span,
                }),
                Err(diagnostic) => diagnostics.push(Located { file: *file, diagnostic }),
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
            if declared[i].sig.name != declared[j].sig.name || !structures_overlap(&declared[i].sig, &declared[j].sig) {
                continue;
            }
            if let Some(message) = contract_mismatch(&declared[j].sig, declared[j].admit, &declared[i].sig, Some(declared[i].admit)) {
                diagnostics.push(Located { file: declared[i].file, diagnostic: Diagnostic::new(declared[i].name_span, message) });
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
            families.push(ContractFamily { name: declared[i].sig.name.clone(), bodies: Vec::new(), contracts: Vec::new(), lowerings: Vec::new(), export: false });
            by_name.entry(declared[i].sig.name.clone()).or_default().push(families.len() - 1);
            families.len() - 1
        });
        declared[i].family = family;
        let id = DefId(i as u32);
        match declared[i].kind {
            DefKind::Body => families[family].bodies.push(id),
            _ => families[family].contracts.push(id),
        }
        families[family].export |= declared[i].export;
    }

    // Lowerings attach to the family their restated signature overlaps (long form) or to
    // every same-name family (short form).
    for (file, scope, parsed) in files {
        for decl in &parsed.decls {
            let ast::Decl::Lower(l) = decl else { continue };
            let Some(named) = by_name.get(&l.name.name).cloned() else {
                diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, format!("`{}` is not declared; a lowering implements a declared function contract", l.name.name)) });
                continue;
            };
            let target = l.target.name.clone();
            if intrinsics::table(&target).is_none() {
                diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.target.span, format!("`{target}` is not a known target")) });
                continue;
            }
            let (kind, body) = match &l.implementation {
                ast::LowerImpl::Body(b) => (DefKind::Lower { target }, Some(b)),
                ast::LowerImpl::Portable => (DefKind::Adopt { target }, None),
            };
            let mut attach: Vec<(usize, Sig, Vec<(String, Elem)>)> = Vec::new();
            match &l.signature {
                Some(signature) => {
                    let sig = match signature_of(&l.name.name, signature, &l.predicates) {
                        Ok(sig) => sig,
                        Err(diagnostic) => {
                            diagnostics.push(Located { file: *file, diagnostic });
                            continue;
                        }
                    };
                    let matching: Vec<(usize, usize)> = named
                        .iter()
                        .filter_map(|family| {
                            let members = families[*family].bodies.iter().chain(&families[*family].contracts);
                            members.map(|id| id.0 as usize).find(|member| structures_overlap(&declared[*member].sig, &sig)).map(|member| (*family, member))
                        })
                        .collect();
                    match matching.as_slice() {
                        [] => diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, format!("no definition of `{}` has this parameter structure (kinds, ranks, element types); a lowering restates the contract it implements", l.name.name)) }),
                        [(family, member)] => {
                            let contract = &declared[*member];
                            if let Some(message) = contract_mismatch(&contract.sig, contract.admit, &sig, None) {
                                diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, message) });
                            }
                            let bindings = elem_bindings(&contract.sig, &sig);
                            attach.push((*family, sig, bindings));
                        }
                        _ => diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, format!("this lowering overlaps several disjoint contract families of `{}`; restate one family's parameter structure", l.name.name)) }),
                    }
                }
                None => {
                    if body.is_some() {
                        diagnostics.push(Located { file: *file, diagnostic: Diagnostic::new(l.name.span, "a lowering with a body restates the signature it implements") });
                        continue;
                    }
                    for family in &named {
                        let Some(member) = families[*family].bodies.first().or(families[*family].contracts.first()) else { continue };
                        let mut sig = declared[member.0 as usize].sig.clone();
                        sig.predicates.clear();
                        let mut failed = false;
                        for e in &l.predicates {
                            if let Err(diagnostic) = predicates_of(e, &sig.shape_params, &mut sig.predicates) {
                                diagnostics.push(Located { file: *file, diagnostic });
                                failed = true;
                            }
                        }
                        if !failed {
                            attach.push((*family, sig, Vec::new()));
                        }
                    }
                }
            }
            for (family, sig, elem_bindings) in attach {
                families[family].lowerings.push(DefId(declared.len() as u32));
                declared.push(Declared { sig, kind: kind.clone(), family, admit: false, export: false, elem_bindings, body, file: *file, scope: scope.clone(), span: l.span, name_span: l.name.span });
            }
        }
    }
    // A lowering inherits the admitted contract of the family it implements.
    for i in 0..declared.len() {
        if declared[i].kind.target().is_some() {
            let family = &families[declared[i].family];
            declared[i].admit = family.bodies.iter().chain(&family.contracts).any(|id| declared[id.0 as usize].admit);
        }
    }
    Resolved { declared, families, by_name }
}

/// Every exported family needs an explicit lowering or adoption on every requested target.
pub(crate) fn coverage(resolved: &Resolved, targets: &[String], diagnostics: &mut Vec<Located>) {
    for family in resolved.families.iter().filter(|f| f.export) {
        let Some(entry) = family.bodies.iter().chain(&family.contracts).map(|id| &resolved.declared[id.0 as usize]).find(|d| d.export) else { continue };
        for target in targets {
            let covered = family.lowerings.iter().any(|id| resolved.declared[id.0 as usize].kind.target() == Some(target.as_str()));
            if !covered {
                diagnostics.push(Located { file: entry.file, diagnostic: Diagnostic::new(entry.name_span, format!("exported `{}` has no `lower {} for {target}` (a body or `= portable`); an exported entry needs explicit coverage on every requested target", family.name, family.name)) });
            }
        }
    }
}
