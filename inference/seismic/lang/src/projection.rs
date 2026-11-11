//! Projections: a backend's consumer forms of projecting activation rows by
//! weights, written once and taken by every entry that projects.
//!
//! # Decision
//!
//! A form is two things: a property of the weight (whether the backend's
//! producer delivers its operand, decided in [`crate::forms`]) and a
//! pipeline (its parameter, admission, shape guards, scratch and launches).
//! The first reaches every caller through `form(W, NAME)`; the second was
//! restated by each entry. A projection owns the second, in the language,
//! so that it is checked, tuned and formed exactly as a hand-written native
//! declaration is:
//!
//! - `projection NAME for TARGET:` declares the forms over formals: `M`
//!   activation rows, `K` columns, `N` rows of each weight, `WEIGHTS` the
//!   weights one activation row serves, and `W` a weight inside a
//!   quantifier. A `let` names an expression over them, per variant where
//!   it carries `for VARIANT`, and so may any parameter, `where` line,
//!   scratch buffer or launch.
//! - Admission quantifies over the weights: `each(W, C)` is C of every
//!   weight, `any(W, C)` of some weight, and `agree(W, E)` that every
//!   weight's E equals the first's; `bound(W)` is a weight's packing-bound
//!   extent. Scratch in an `each W:` block is one buffer per weight.
//! - An entry's native declaration takes a projection with one line,
//!   `project NAME VARIANT (WEIGHT, ..)`, naming its weight ports. Each port
//!   is `<stem>_weight: tensor[N, K] E`, its bound port (if any)
//!   `<stem>_bound: tensor[B] f32`, and the entry returns `[M, N]`: the
//!   formals are read from the signature, never restated. A projection
//!   launch `NAME` is the entry's kernel `<entry>_NAME`, and a per-weight
//!   buffer `NAME` is `<stem>_NAME`.
//! - [`expand`] rewrites every such declaration into the ordinary native
//!   declaration it stands for, before checking. The checker, coverage,
//!   the tuner's parameter addresses and formation see only that
//!   declaration, so moving an entry onto a projection changes none of
//!   them: its parameters follow the entry's, its conditions follow its
//!   `where`, its scratch sits at the `project` line, and its launches keep
//!   their order, an entry launch placed by `before LAUNCH`.
//!
//! A variant (`paired`, `single`) is fixed by the entry and folds away:
//! formals and terms substitute, literal arithmetic and comparisons
//! evaluate, `x * 1` and `x / 1` reduce, and a condition that becomes true
//! drops out of its conjunction. The expansion is therefore exactly the
//! text an entry would write by hand.
//!
//! The native libraries own the matching kernel bodies, per backend
//! (`metal/lib/projection/operator.h`); an entry's source keeps only its
//! prologue, epilogue and weights.

use std::collections::{BTreeMap, BTreeSet};

use crate::checked::DiagnosticRule;
use crate::span::{Diagnostic, Span};
use crate::syntax::ast::*;

/// The variants an entry projects by: the number of weights one activation
/// row serves.
const VARIANTS: [(&str, usize); 2] = [("paired", 2), ("single", 1)];

/// Rewrite every native declaration of the parsed files that projects into
/// the declaration it stands for, and remove the projections. The
/// diagnostics name the file (its index in `parsed`'s numbering) they are
/// in.
pub(crate) fn expand(parsed: &mut [(usize, File)]) -> Vec<(usize, Diagnostic)> {
    let mut diagnostics = Vec::new();
    let mut projections: BTreeMap<String, (usize, ProjectionDecl)> = BTreeMap::new();
    for (file, ast) in parsed.iter_mut() {
        let mut kept = Vec::with_capacity(ast.decls.len());
        for decl in std::mem::take(&mut ast.decls) {
            match decl {
                Decl::Projection(projection) => {
                    if let Err(diagnostic) = gates(&projection) {
                        diagnostics.push((*file, diagnostic));
                    }
                    if projections.contains_key(&projection.name.name) {
                        diagnostics.push((
                            *file,
                            error(
                                projection.name.span,
                                format!("projection `{}` is declared twice", projection.name.name),
                            ),
                        ));
                    }
                    projections.insert(projection.name.name.clone(), (*file, projection));
                }
                other => kept.push(other),
            }
        }
        ast.decls = kept;
    }
    let mut signatures: BTreeMap<String, Vec<Signature>> = BTreeMap::new();
    for (_, ast) in parsed.iter() {
        for decl in &ast.decls {
            if let Decl::Fn(function) = decl {
                signatures
                    .entry(function.name.name.clone())
                    .or_default()
                    .push(function.signature.clone());
            }
        }
    }
    for (file, ast) in parsed.iter_mut() {
        for decl in &mut ast.decls {
            let Decl::Native(native) = decl else {
                continue;
            };
            let result = match native.project.take() {
                Some(project) => {
                    expand_native(native, project, *file, &projections, &signatures)
                }
                None => match native.launches.iter().find_map(|launch| launch.before.as_ref()) {
                    Some(before) => Err(error(
                        before.span,
                        "`before` places a launch among a projection's; this declaration does not project",
                    )),
                    None => Ok(()),
                },
            };
            if let Err(diagnostic) = result {
                diagnostics.push((*file, diagnostic));
            }
        }
    }
    diagnostics
}

fn error(span: Span, message: impl Into<String>) -> Diagnostic {
    Diagnostic::with_rule(DiagnosticRule::NativeDeclaration, span, message)
}

/// Every `for VARIANT` of a projection names a variant.
fn gates(projection: &ProjectionDecl) -> Result<(), Diagnostic> {
    let mut variants = Vec::new();
    variants.extend(projection.terms.iter().map(|term| &term.variant));
    variants.extend(projection.params.iter().map(|param| &param.variant));
    variants.extend(projection.constraints.iter().map(|constraint| &constraint.variant));
    for scratch in &projection.scratch {
        match scratch {
            ProjectionScratchDecl::One(one) => variants.push(&one.variant),
            ProjectionScratchDecl::Each(each) => variants.extend(each.iter().map(|one| &one.variant)),
        }
    }
    for launch in &projection.launches {
        variants.push(&launch.variant);
        variants.extend(launch.param_variants.iter());
    }
    for variant in variants.into_iter().flatten() {
        if !VARIANTS.iter().any(|(name, _)| *name == variant.name) {
            return Err(error(
                variant.span,
                format!("`{}` is not a projection variant (`paired`, `single`)", variant.name),
            ));
        }
    }
    Ok(())
}

/// One weight an entry projects by.
struct Weight {
    /// Its element parameter: what `W` stands for.
    element: Ident,
    /// Its port's name without `_weight`.
    stem: String,
    /// Its bound port's extent (`bound(W)`).
    bound: Option<Ident>,
}

/// One projection instantiated for one entry.
struct Instance<'a> {
    entry: &'a str,
    variant: &'a str,
    m: Ident,
    k: Ident,
    n: Ident,
    weights: Vec<Weight>,
    /// The projection's terms in force for the variant.
    terms: BTreeMap<String, Expr>,
    /// Every term name of the projection, in force or not.
    named: BTreeSet<String>,
    /// Where the projection's text sits: its spans where it shares the
    /// entry's file, else the `project` line.
    same_file: bool,
    at: Span,
}

fn expand_native(
    native: &mut NativeDecl,
    project: NativeProjectDecl,
    file: usize,
    projections: &BTreeMap<String, (usize, ProjectionDecl)>,
    signatures: &BTreeMap<String, Vec<Signature>>,
) -> Result<(), Diagnostic> {
    let Some((projection_file, projection)) = projections.get(&project.projection.name) else {
        return Err(error(
            project.projection.span,
            format!("no projection `{}`", project.projection.name),
        ));
    };
    if projection.target.name != native.target.name {
        return Err(error(
            project.projection.span,
            format!(
                "projection `{}` is for `{}`, not `{}`",
                projection.name.name, projection.target.name, native.target.name
            ),
        ));
    }
    let Some(&(variant, count)) = VARIANTS.iter().find(|(name, _)| *name == project.variant.name) else {
        return Err(error(
            project.variant.span,
            format!("`{}` is not a projection variant (`paired`, `single`)", project.variant.name),
        ));
    };
    if project.weights.len() != count {
        return Err(error(
            project.span,
            format!("a `{variant}` projection takes {count} weight(s), not {}", project.weights.len()),
        ));
    }
    if native.repeat.is_some() {
        return Err(error(project.span, "a projecting declaration has no `repeat` block"));
    }
    let signature = match signatures.get(&native.function.name).map(Vec::as_slice) {
        Some([signature]) => signature,
        _ => {
            return Err(error(
                project.span,
                format!("`{}` is not one portable function to project", native.function.name),
            ));
        }
    };
    let dims = |ty: &TypeExpr| -> Option<(Vec<Ident>, Ident)> {
        let TypeKind::Shaped { shape, elem, .. } = &ty.kind else {
            return None;
        };
        let names = shape
            .iter()
            .map(|dim| match &dim.kind {
                ExprKind::Name(name) => Some(name.clone()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        Some((names, elem.clone()))
    };
    let port = |name: &str| signature.params.iter().find(|param| param.name.name == name);
    let mut weights = Vec::new();
    let mut shape: Option<(Ident, Ident)> = None;
    for weight in &project.weights {
        let Some(param) = port(&weight.name) else {
            return Err(error(weight.span, format!("`{}` is not a port of `{}`", weight.name, native.function.name)));
        };
        let Some(stem) = weight.name.strip_suffix("_weight") else {
            return Err(error(weight.span, format!("a weight port is `<stem>_weight`, not `{}`", weight.name)));
        };
        let Some((dims_of, element)) = dims(&param.ty).filter(|(dims_of, _)| dims_of.len() == 2) else {
            return Err(error(weight.span, format!("weight `{}` is not a `tensor[N, K]`", weight.name)));
        };
        let (n, k) = (dims_of[0].clone(), dims_of[1].clone());
        if let Some((first_n, first_k)) = &shape {
            if first_n.name != n.name || first_k.name != k.name {
                return Err(error(weight.span, "the weights of a projection share `N` and `K`"));
            }
        } else {
            shape = Some((n, k));
        }
        let bound = match port(&format!("{stem}_bound")) {
            Some(bound) => match dims(&bound.ty) {
                Some((extent, _)) if extent.len() == 1 => Some(extent[0].clone()),
                _ => return Err(error(weight.span, format!("`{stem}_bound` is not a `tensor[B]`"))),
            },
            None => None,
        };
        weights.push(Weight {
            element,
            stem: stem.to_owned(),
            bound,
        });
    }
    let (n, k) = shape.expect("a variant takes at least one weight");
    let m = match signature.result.as_ref().and_then(|result| dims(result)) {
        Some((result, _)) if result.len() == 2 && result[1].name == n.name => result[0].clone(),
        _ => {
            return Err(error(
                project.span,
                format!("a projecting entry returns `[M, N]`, `N` its weights' `{}`", n.name),
            ));
        }
    };
    let mut instance = Instance {
        entry: &native.function.name,
        variant,
        m,
        k,
        n,
        weights,
        terms: BTreeMap::new(),
        named: projection.terms.iter().map(|term| term.item.name.name.clone()).collect(),
        same_file: *projection_file == file,
        at: project.span,
    };
    for term in &projection.terms {
        if !instance.takes(&term.variant) {
            continue;
        }
        if instance.formal(&term.item.name.name) {
            return Err(error(instance.span(term.item.name.span), format!("`{}` is a projection formal", term.item.name.name)));
        }
        let value = instance.expr(&term.item.value)?;
        instance.terms.insert(term.item.name.name.clone(), value);
    }

    let mut params = std::mem::take(&mut native.params);
    for param in &projection.params {
        if instance.takes(&param.variant) {
            params.push(instance.param(&param.item)?);
        }
    }
    native.params = params;

    let mut conjuncts = Vec::new();
    for constraint in &projection.constraints {
        if instance.takes(&constraint.variant) {
            flatten(instance.expr(&constraint.item)?, BinaryOp::And, &mut conjuncts);
        }
    }
    for conjunct in conjuncts {
        if matches!(conjunct.kind, ExprKind::Bool(true)) {
            continue;
        }
        native.constraint = Some(match native.constraint.take() {
            None => conjunct,
            Some(constraint) => binary(BinaryOp::And, constraint, conjunct),
        });
    }

    let mut scratch = Vec::new();
    for item in &projection.scratch {
        match item {
            ProjectionScratchDecl::One(one) => {
                if instance.takes(&one.variant) {
                    scratch.push(instance.scratch(&one.item, None)?);
                }
            }
            ProjectionScratchDecl::Each(each) => {
                for weight in &instance.weights {
                    for one in each {
                        if instance.takes(&one.variant) {
                            scratch.push(instance.scratch(&one.item, Some(weight))?);
                        }
                    }
                }
            }
        }
    }
    let after = native.scratch.split_off(project.scratch_before.min(native.scratch.len()));
    native.scratch.extend(scratch);
    native.scratch.extend(after);

    let own = std::mem::take(&mut native.launches);
    let mut placed = vec![false; own.len()];
    let mut launches = Vec::new();
    for launch in &projection.launches {
        if !instance.takes(&launch.variant) {
            continue;
        }
        for (ordinal, entry) in own.iter().enumerate() {
            if entry.before.as_ref().is_some_and(|before| before.name == launch.launch.kernel.name) {
                placed[ordinal] = true;
                launches.push(NativeLaunchDecl {
                    before: None,
                    ..entry.clone()
                });
            }
        }
        launches.push(instance.launch(launch)?);
    }
    for (ordinal, entry) in own.into_iter().enumerate() {
        match &entry.before {
            Some(before) if !placed[ordinal] => {
                return Err(error(
                    before.span,
                    format!(
                        "projection `{}` has no `{variant}` launch `{}`",
                        projection.name.name, before.name
                    ),
                ));
            }
            Some(_) => {}
            None => launches.push(entry),
        }
    }
    native.launches = launches;
    Ok(())
}

/// `lhs op rhs`, spanning both.
fn binary(op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr {
        span: lhs.span.to(rhs.span),
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
    }
}

/// The operands of `expr`'s chain of `op` (`and`, `or`), left to right.
fn flatten(expr: Expr, op: BinaryOp, out: &mut Vec<Expr>) {
    match expr.kind {
        ExprKind::Binary { op: inner, lhs, rhs } if inner == op => {
            flatten(*lhs, op, out);
            flatten(*rhs, op, out);
        }
        kind => out.push(Expr { kind, span: expr.span }),
    }
}

/// `operands` joined by `op` (`and`, `or`) from the left; an empty `and`
/// is true and an empty `or` false.
fn chain(op: BinaryOp, operands: Vec<Expr>, span: Span) -> Expr {
    operands
        .into_iter()
        .reduce(|lhs, rhs| binary(op, lhs, rhs))
        .unwrap_or(Expr {
            kind: ExprKind::Bool(op == BinaryOp::And),
            span,
        })
}

impl Instance<'_> {
    fn takes(&self, variant: &Option<Ident>) -> bool {
        variant.as_ref().is_none_or(|variant| variant.name == self.variant)
    }

    fn formal(&self, name: &str) -> bool {
        matches!(name, "M" | "K" | "N" | "W" | "WEIGHTS")
    }

    fn span(&self, span: Span) -> Span {
        if self.same_file { span } else { self.at }
    }

    fn ident(&self, ident: &Ident) -> Ident {
        Ident {
            name: ident.name.clone(),
            span: self.span(ident.span),
        }
    }

    /// `expr` with the formals and terms substituted, folded.
    fn expr(&self, expr: &Expr) -> Result<Expr, Diagnostic> {
        Ok(fold(self.substitute(expr, None)?))
    }

    fn substitute(&self, expr: &Expr, weight: Option<&Weight>) -> Result<Expr, Diagnostic> {
        let span = self.span(expr.span);
        let kind = match &expr.kind {
            ExprKind::Name(name) => match name.name.as_str() {
                "M" => ExprKind::Name(Ident { name: self.m.name.clone(), span }),
                "K" => ExprKind::Name(Ident { name: self.k.name.clone(), span }),
                "N" => ExprKind::Name(Ident { name: self.n.name.clone(), span }),
                "WEIGHTS" => ExprKind::Int(self.weights.len() as u64),
                "W" => match weight {
                    Some(weight) => ExprKind::Name(Ident {
                        name: weight.element.name.clone(),
                        span,
                    }),
                    None => {
                        return Err(error(
                            self.span(expr.span),
                            "`W` is a weight only in `each`, `any`, `agree` or an `each W:` block",
                        ));
                    }
                },
                other => match self.terms.get(other) {
                    Some(value) => return Ok(value.clone()),
                    None if self.named.contains(other) => {
                        return Err(error(
                            self.span(expr.span),
                            format!("`{other}` has no `{}` value", self.variant),
                        ));
                    }
                    None => ExprKind::Name(Ident { name: other.to_owned(), span }),
                },
            },
            ExprKind::Call { callee, bindings, args } => {
                let quantifier = match &callee.kind {
                    ExprKind::Name(name) if bindings.is_empty() => name.name.as_str(),
                    _ => "",
                };
                match quantifier {
                    "each" | "any" | "agree" => return self.quantify(quantifier, args, weight, expr.span),
                    "bound" => {
                        let (Some(weight), [Arg { name: None, value }]) = (weight, args.as_slice()) else {
                            return Err(error(self.span(expr.span), "`bound(W)` reads the extent of a weight `W`"));
                        };
                        if !matches!(&value.kind, ExprKind::Name(name) if name.name == "W") {
                            return Err(error(self.span(expr.span), "`bound(W)` reads the extent of a weight `W`"));
                        }
                        let Some(bound) = &weight.bound else {
                            return Err(error(
                                self.span(expr.span),
                                format!("weight `{}_weight` has no `{}_bound` port", weight.stem, weight.stem),
                            ));
                        };
                        ExprKind::Name(Ident { name: bound.name.clone(), span })
                    }
                    _ => ExprKind::Call {
                        callee: Box::new(self.substitute(callee, weight)?),
                        bindings: bindings
                            .iter()
                            .map(|(name, value)| Ok((self.ident(name), self.substitute(value, weight)?)))
                            .collect::<Result<_, Diagnostic>>()?,
                        args: args
                            .iter()
                            .map(|arg| {
                                Ok(Arg {
                                    name: arg.name.as_ref().map(|name| self.ident(name)),
                                    value: self.substitute(&arg.value, weight)?,
                                })
                            })
                            .collect::<Result<_, Diagnostic>>()?,
                    },
                }
            }
            ExprKind::Attr { base, name } => ExprKind::Attr {
                base: Box::new(self.substitute(base, weight)?),
                name: self.ident(name),
            },
            ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
                op: *op,
                expr: Box::new(self.substitute(inner, weight)?),
            },
            ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
                op: *op,
                lhs: Box::new(self.substitute(lhs, weight)?),
                rhs: Box::new(self.substitute(rhs, weight)?),
            },
            ExprKind::Tuple(items) => ExprKind::Tuple(
                items
                    .iter()
                    .map(|item| self.substitute(item, weight))
                    .collect::<Result<_, _>>()?,
            ),
            kind @ (ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Inf | ExprKind::Bool(_)) => kind.clone(),
            ExprKind::Range { .. } | ExprKind::Tensor { .. } | ExprKind::Index { .. } => {
                return Err(error(self.span(expr.span), "not an expression of a native declaration"));
            }
        };
        Ok(Expr { kind, span })
    }

    /// `each(W, C)`, `any(W, C)` or `agree(W, E)` over the weights.
    fn quantify(
        &self,
        quantifier: &str,
        args: &[Arg],
        weight: Option<&Weight>,
        span: Span,
    ) -> Result<Expr, Diagnostic> {
        let [Arg { name: None, value: formal }, Arg { name: None, value: body }] = args else {
            return Err(error(self.span(span), format!("`{quantifier}(W, ..)` takes the weight `W` and one expression")));
        };
        if !matches!(&formal.kind, ExprKind::Name(name) if name.name == "W") {
            return Err(error(self.span(formal.span), format!("`{quantifier}` ranges over the weights, `W`")));
        }
        if weight.is_some() {
            return Err(error(self.span(span), "weight quantifiers do not nest"));
        }
        let each = self
            .weights
            .iter()
            .map(|weight| self.substitute(body, Some(weight)))
            .collect::<Result<Vec<_>, _>>()?;
        let span = self.span(span);
        Ok(match quantifier {
            "each" => chain(BinaryOp::And, each, span),
            "any" => chain(BinaryOp::Or, each, span),
            _ => {
                let first = each[0].clone();
                let pairs = each[1..]
                    .iter()
                    .map(|other| binary(BinaryOp::Eq, first.clone(), other.clone()))
                    .collect();
                chain(BinaryOp::And, pairs, span)
            }
        })
    }

    fn param(&self, param: &NativeParamDecl) -> Result<NativeParamDecl, Diagnostic> {
        Ok(NativeParamDecl {
            name: self.ident(&param.name),
            arithmetic: param.arithmetic,
            form: param.form,
            values: param.values.clone(),
            default: match &param.default {
                NativeDefaultDecl::Value(value) => NativeDefaultDecl::Value(self.expr(value)?),
                NativeDefaultDecl::Least(condition) => NativeDefaultDecl::Least(self.expr(condition)?),
                NativeDefaultDecl::Greatest(condition) => NativeDefaultDecl::Greatest(self.expr(condition)?),
            },
            span: self.span(param.span),
        })
    }

    fn optional(&self, expr: &Option<Expr>, weight: Option<&Weight>) -> Result<Option<Expr>, Diagnostic> {
        expr.as_ref()
            .map(|expr| Ok(fold(self.substitute(expr, weight)?)))
            .transpose()
    }

    fn scratch(&self, scratch: &NativeScratchDecl, weight: Option<&Weight>) -> Result<NativeScratchDecl, Diagnostic> {
        let name = match weight {
            Some(weight) => Ident {
                name: format!("{}_{}", weight.stem, scratch.name.name),
                span: self.span(scratch.name.span),
            },
            None => self.ident(&scratch.name),
        };
        Ok(NativeScratchDecl {
            name,
            bytes: fold(self.substitute(&scratch.bytes, weight)?),
            sync: scratch.sync,
            when: self.optional(&scratch.when, weight)?,
            span: self.span(scratch.span),
        })
    }

    fn launch(&self, launch: &ProjectionLaunchDecl) -> Result<NativeLaunchDecl, Diagnostic> {
        let declared = &launch.launch;
        let mut params = Vec::new();
        for (param, variant) in declared.params.iter().zip(&launch.param_variants) {
            if self.takes(variant) {
                params.push(self.param(param)?);
            }
        }
        let three = |exprs: &[Expr; 3]| -> Result<[Expr; 3], Diagnostic> {
            Ok([self.expr(&exprs[0])?, self.expr(&exprs[1])?, self.expr(&exprs[2])?])
        };
        Ok(NativeLaunchDecl {
            kernel: Ident {
                name: format!("{}_{}", self.entry, declared.kernel.name),
                span: self.span(declared.kernel.span),
            },
            before: None,
            resident: declared.resident.as_ref().map(|barrier| self.ident(barrier)),
            when: self.optional(&declared.when, None)?,
            params,
            constraint: self.optional(&declared.constraint, None)?,
            reads: declared.reads.iter().map(|read| self.ident(read)).collect(),
            records: declared.records.as_ref().map(|records| {
                records
                    .iter()
                    .map(|record| match record {
                        NativeRecordDecl::Parameter(name) => NativeRecordDecl::Parameter(self.ident(name)),
                        NativeRecordDecl::Result { ordinal, span } => NativeRecordDecl::Result {
                            ordinal: *ordinal,
                            span: self.span(*span),
                        },
                    })
                    .collect()
            }),
            threadgroups: three(&declared.threadgroups)?,
            grid: declared.grid.as_ref().map(|grid| self.ident(grid)),
            conditions: declared
                .conditions
                .iter()
                .map(|condition| {
                    Ok(NativeConditionDecl {
                        encoded: condition.encoded,
                        word: self.ident(&condition.word),
                        index: self.optional(&condition.index, None)?,
                    })
                })
                .collect::<Result<_, Diagnostic>>()?,
            threads_per_threadgroup: three(&declared.threads_per_threadgroup)?,
            shared_bytes: self.optional(&declared.shared_bytes, None)?,
            span: self.span(declared.span),
        })
    }
}

/// `expr` with its constant parts evaluated: literal arithmetic and
/// comparisons, `x * 1`, `1 * x` and `x / 1`, `not` of a literal, and
/// conjunctions and disjunctions without their decided operands (flat,
/// from the left).
fn fold(expr: Expr) -> Expr {
    let span = expr.span;
    let kind = match expr.kind {
        ExprKind::Binary { op: op @ (BinaryOp::And | BinaryOp::Or), lhs, rhs } => {
            let mut operands = Vec::new();
            flatten(fold(*lhs), op, &mut operands);
            flatten(fold(*rhs), op, &mut operands);
            let absorbing = op == BinaryOp::Or;
            if operands.iter().any(|operand| matches!(operand.kind, ExprKind::Bool(value) if value == absorbing)) {
                return Expr { kind: ExprKind::Bool(absorbing), span };
            }
            operands.retain(|operand| !matches!(operand.kind, ExprKind::Bool(_)));
            let mut joined = chain(op, operands, span);
            joined.span = span;
            return joined;
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let (lhs, rhs) = (fold(*lhs), fold(*rhs));
            match (&lhs.kind, &rhs.kind) {
                (&ExprKind::Int(a), &ExprKind::Int(b)) => match op {
                    BinaryOp::Add => a.checked_add(b).map(ExprKind::Int),
                    BinaryOp::Sub => a.checked_sub(b).map(ExprKind::Int),
                    BinaryOp::Mul => a.checked_mul(b).map(ExprKind::Int),
                    BinaryOp::Div => a.checked_div(b).map(ExprKind::Int),
                    BinaryOp::Rem => a.checked_rem(b).map(ExprKind::Int),
                    BinaryOp::Eq => Some(ExprKind::Bool(a == b)),
                    BinaryOp::Ne => Some(ExprKind::Bool(a != b)),
                    BinaryOp::Lt => Some(ExprKind::Bool(a < b)),
                    BinaryOp::Le => Some(ExprKind::Bool(a <= b)),
                    BinaryOp::Gt => Some(ExprKind::Bool(a > b)),
                    BinaryOp::Ge => Some(ExprKind::Bool(a >= b)),
                    _ => None,
                },
                (_, ExprKind::Int(1)) if matches!(op, BinaryOp::Mul | BinaryOp::Div) => return lhs,
                (ExprKind::Int(1), _) if op == BinaryOp::Mul => return rhs,
                _ => None,
            }
            .unwrap_or(ExprKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        }
        ExprKind::Unary { op, expr: inner } => match (op, fold(*inner)) {
            (UnaryOp::Not, Expr { kind: ExprKind::Bool(value), .. }) => ExprKind::Bool(!value),
            (op, inner) => ExprKind::Unary {
                op,
                expr: Box::new(inner),
            },
        },
        ExprKind::Call { callee, bindings, args } => ExprKind::Call {
            callee,
            bindings: bindings.into_iter().map(|(name, value)| (name, fold(value))).collect(),
            args: args
                .into_iter()
                .map(|arg| Arg {
                    name: arg.name,
                    value: fold(arg.value),
                })
                .collect(),
        },
        kind => kind,
    };
    Expr { kind, span }
}

#[cfg(test)]
mod tests {
    use crate::checked::{check_source, NativeImplementation, SourceFile, SourceSet};
    use crate::registry::BackendName;

    /// A paired and a single entry, each with a bound port per weight.
    const FUNCTIONS: &str = "fn pair[O, K, N, GB, UB](x: &tensor[O, K] f32, gate_weight: &tensor[N, K] GW, up_weight: &tensor[N, K] UW, gate_bound: &tensor[GB] f32, up_bound: &tensor[UB] f32) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\nfn one[O, K, N, DB](x: &tensor[O, K] f32, down_weight: &tensor[N, K] DW, down_bound: &tensor[DB] f32) -> tensor[O, N] f32:\n    let mut result = tensor[O, N] f32\n    parallel for row in 0..O:\n        parallel for col in 0..N:\n            result[row, col] = x[row, 0]\n    return result\n\n";

    const PROJECTION: &str = "projection lanes for metal:\n    let TOKENS = 128 / WEIGHTS\n    let SPLITS = 1 for paired\n    let SPLITS = SPLIT for single\n    params (arithmetic SPLIT in [1, 2] default 1 for single, form WIDE in [0, 1] default 0)\n    where (WIDE == 0 or (SPLITS == 1 and K % (64 / WEIGHTS) == 0 and each(W, form(W, INT8) == 1) and agree(W, form(W, INT8_INTERLEAVED))))\n    where (WIDE == 0 or any(W, bound(W) > 0 and form(W, PACK) == 1))\n    scratch partials bytes (SPLIT * M * N * 4) when M > 4 and SPLIT > 1 for single\n    scratch packed bytes (ceil_div(M, TOKENS) * TOKENS * K) when WIDE == 1\n    each W:\n        scratch factors bytes (N * (K / 32) * 4) when WIDE == 1\n        scratch scales bytes (N * 4) when WIDE == 1\n    launch gemv when M < 2:\n        params (ROWS in [1, 2] default 1 for paired, ROWS in [2, 1] default 2 for single)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (ROWS * (128 * WEIGHTS))\n    launch reduce for single when M > 4 and SPLIT > 1:\n        threadgroups (ceil_div(M * N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch tiles when M >= 2:\n        threadgroups (ceil_div(N, 32 / WEIGHTS), ceil_div(M, TOKENS) * (TOKENS / 4), SPLITS)\n        threads_per_threadgroup (128, 1, 1)\n\n";

    const PROJECTED: &str = "native pair for metal from \"pair.metal\":\n    static (K, N, GB, UB)\n    where N > 0\n    scratch staged bytes (O * K)\n    project lanes paired (gate_weight, up_weight)\n    launch pair_stage before tiles when O >= 2:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n\nnative one for metal from \"one.metal\":\n    static (K, N, DB)\n    project lanes single (down_weight)\n    scratch staged bytes (O * K)\n    launch one_stage before gemv when O >= 2:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n";

    /// What `PROJECTED` stands for, written by hand.
    const WRITTEN: &str = "native pair for metal from \"pair.metal\":\n    static (K, N, GB, UB)\n    params (form WIDE in [0, 1] default 0)\n    where N > 0 and (WIDE == 0 or (K % 32 == 0 and form(GW, INT8) == 1 and form(UW, INT8) == 1 and form(GW, INT8_INTERLEAVED) == form(UW, INT8_INTERLEAVED))) and (WIDE == 0 or (GB > 0 and form(GW, PACK) == 1) or (UB > 0 and form(UW, PACK) == 1))\n    scratch staged bytes (O * K)\n    scratch packed bytes (ceil_div(O, 64) * 64 * K) when WIDE == 1\n    scratch gate_factors bytes (N * (K / 32) * 4) when WIDE == 1\n    scratch gate_scales bytes (N * 4) when WIDE == 1\n    scratch up_factors bytes (N * (K / 32) * 4) when WIDE == 1\n    scratch up_scales bytes (N * 4) when WIDE == 1\n    launch pair_gemv when O < 2:\n        params (ROWS in [1, 2] default 1)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (ROWS * 256)\n    launch pair_stage when O >= 2:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch pair_tiles when O >= 2:\n        threadgroups (ceil_div(N, 16), ceil_div(O, 64) * 16, 1)\n        threads_per_threadgroup (128, 1, 1)\n\nnative one for metal from \"one.metal\":\n    static (K, N, DB)\n    params (arithmetic SPLIT in [1, 2] default 1, form WIDE in [0, 1] default 0)\n    where (WIDE == 0 or (SPLIT == 1 and K % 64 == 0 and form(DW, INT8) == 1)) and (WIDE == 0 or DB > 0 and form(DW, PACK) == 1)\n    scratch partials bytes (SPLIT * O * N * 4) when O > 4 and SPLIT > 1\n    scratch packed bytes (ceil_div(O, 128) * 128 * K) when WIDE == 1\n    scratch down_factors bytes (N * (K / 32) * 4) when WIDE == 1\n    scratch down_scales bytes (N * 4) when WIDE == 1\n    scratch staged bytes (O * K)\n    launch one_stage when O >= 2:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch one_gemv when O < 2:\n        params (ROWS in [2, 1] default 2)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n        shared_bytes (ROWS * 128)\n    launch one_reduce when O > 4 and SPLIT > 1:\n        threadgroups (ceil_div(O * N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch one_tiles when O >= 2:\n        threadgroups (ceil_div(N, 32), ceil_div(O, 128) * 32, SPLIT)\n        threads_per_threadgroup (128, 1, 1)\n";

    fn implementations(natives: &str) -> Vec<NativeImplementation> {
        let mut sources = SourceSet::default();
        sources.push(SourceFile {
            path: "ops.seismic".to_owned(),
            text: format!("{FUNCTIONS}{natives}"),
        });
        let module = crate::native_source::check_fixture(sources, &[]).unwrap_or_else(|error| panic!("{error}"));
        ["pair", "one"]
            .into_iter()
            .map(|name| {
                let entry = module.entry_named(name).unwrap();
                module.native_implementation(entry, BackendName::Metal).unwrap().clone()
            })
            .collect()
    }

    /// An entry that projects checks to exactly the implementation it would
    /// write by hand: its parameters, conditions (each weight, some weight,
    /// agreement between weights; folded terms and formals), scratch per
    /// weight at the `project` line, and launches named for the entry in the
    /// projection's order, its own placed `before` them.
    #[test]
    fn a_projecting_entry_checks_to_the_declaration_it_stands_for() {
        let projected = implementations(&format!("{PROJECTION}{PROJECTED}"));
        let mut written = implementations(WRITTEN);
        for (written, projected) in written.iter_mut().zip(&projected) {
            written.entry = projected.entry;
        }
        assert_eq!(projected, written);
    }

    #[test]
    fn a_projection_refuses_what_it_cannot_expand() {
        let refused = |natives: &str, message: &str| {
            let mut sources = SourceSet::default();
            sources.push(SourceFile {
                path: "ops.seismic".to_owned(),
                text: format!("{FUNCTIONS}{PROJECTION}{natives}"),
            });
            let error = check_source(sources).expect_err(message).to_string();
            assert!(error.contains(message), "{error}");
        };
        let native = |project: &str, launch: &str| {
            format!("native pair for metal from \"pair.metal\":\n    {project}\n    launch pair_stage{launch}:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n")
        };
        refused(&native("project lanes paired (gate_weight)", ""), "takes 2 weight(s)");
        refused(&native("project lanes paired (gate_weight, up_weight)", " before reduce"), "no `paired` launch `reduce`");
        refused(&native("project lanes twice (gate_weight, up_weight)", ""), "not a projection variant");
        refused(&native("project lanes paired (gate_weight, x)", ""), "`<stem>_weight`");
        refused(&native("scratch staged bytes (4)", " before gemv"), "does not project");
    }
}
