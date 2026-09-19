use super::{Expression as E, Program, Statement as S, Type};
use seismic_lang::ast::{BinaryOp, UnaryOp};
use std::collections::{HashMap, HashSet};

fn expression(value: &E) -> bool {
    match value {
        E::Integer(..)
        | E::Float(..)
        | E::Variable(..)
        | E::VectorElement { .. }
        | E::Parameter { .. } => true,
        E::Binary(_, a, b, _)
        | E::ShortCircuit {
            left: a, right: b, ..
        } => expression(a) && expression(b),
        E::Unary(_, value, _) | E::Cast(_, value) | E::Bitcast(_, value) => expression(value),
        E::Builtin(_, values, _) | E::Helper(_, values, _) => values.iter().all(expression),
        E::Select(condition, yes, no) | E::EagerSelect(condition, yes, no) => expression(condition) && expression(yes) && expression(no),
        E::Read { index, .. } => expression(index),
        E::Unmapped(..) => false,
    }
}

fn statement(statement: &S) -> bool {
    match statement {
        S::Let { value, .. }
        | S::Assign { value, .. }
        | S::Evaluate(value)
        | S::If(value)
        | S::ReturnIf(value) => expression(value),
        S::Pointer { index, .. } | S::VectorRead { index, .. } => expression(index),
        S::Write { index, value, .. } => expression(index) && expression(value),
        S::For { start, end, .. } => expression(start) && expression(end),
        S::Return(value) => value.as_ref().is_none_or(expression),
        S::MatrixLoad {
            offset, leading, ..
        }
        | S::MatrixStore {
            offset, leading, ..
        } => expression(offset) && expression(leading),
        S::Array { .. }
        | S::Else
        | S::Scope
        | S::End
        | S::FailureStatus
        | S::Barrier
        | S::Fragment { .. }
        | S::MatrixMultiplyAccumulate { .. } => true,
        S::Unmapped(_) => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Root,
    Block,
    Then,
    Else,
}

fn integer(ty: Type) -> bool {
    matches!(ty, Type::I32 | Type::U32 | Type::I64 | Type::U64)
}
fn declared(statement: &S) -> Option<(&str, Option<Type>)> {
    match statement {
        S::Let { name, ty, .. } => Some((name, Some(*ty))),
        S::For { name, .. } => Some((name, Some(Type::I32))),
        S::Array { name, .. }
        | S::Pointer { name, .. }
        | S::VectorRead { name, .. }
        | S::Fragment { name, .. } => Some((name, None)),
        _ => None,
    }
}
fn binding<'a>(
    name: &str,
    scopes: &'a [(Scope, HashMap<String, Option<Type>>)],
    locals: &HashSet<String>,
) -> Result<Option<&'a Option<Type>>, String> {
    if !locals.contains(name) { return Ok(None); }
    scopes.iter().rev().find_map(|(_, names)| names.get(name)).map(Some)
        .ok_or_else(|| format!("local `{name}` is outside its lexical scope"))
}
fn typed(
    value: &E,
    scopes: &[(Scope, HashMap<String, Option<Type>>)],
    locals: &HashSet<String>,
) -> Result<(), String> {
    let check = |value| typed(value, scopes, locals);
    match value {
        E::Variable(name, ty) => {
            if binding(name, scopes, locals)?.copied().flatten().is_some_and(|actual| actual != *ty) {
                return Err(format!("local `{name}` has a mismatched expression type"));
            }
        }
        E::Binary(op, a, b, ty) => {
            check(a)?;
            check(b)?;
            let comparison = matches!(
                op,
                BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
            );
            if comparison && *ty != Type::Bool {
                return Err("comparison does not produce bool".into());
            }
            if !matches!(op, BinaryOp::Shl | BinaryOp::Shr) && a.ty() != b.ty() {
                return Err("binary operands need explicit common type".into());
            }
        }
        E::ShortCircuit { left, right, .. } => {
            check(left)?;
            check(right)?;
            if left.ty() != Type::Bool || right.ty() != Type::Bool {
                return Err("short-circuit operands must be bool".into());
            }
        }
        E::Select(condition, yes, no) | E::EagerSelect(condition, yes, no) => {
            check(condition)?;
            check(yes)?;
            check(no)?;
            if condition.ty() != Type::Bool || yes.ty() != no.ty() {
                return Err("conditional expression has inconsistent types".into());
            }
        }
        E::Unary(op, value, ty) => {
            check(value)?;
            if *op == UnaryOp::Not && (*ty != Type::Bool || value.ty() != Type::Bool) {
                return Err("logical negation must use bool".into());
            }
        }
        E::Cast(_, value) => check(value)?,
        E::Bitcast(ty, value) => {
            check(value)?;
            if ty.bytes() != value.ty().bytes() {
                return Err("bitcast changes storage width".into());
            }
        }
        E::Read { name, index, .. } => {
            binding(name, scopes, locals)?;
            check(index)?;
            if !integer(index.ty()) {
                return Err("memory index must be integral".into());
            }
        }
        E::Builtin(_, args, _) | E::Helper(_, args, _) => {
            for arg in args {
                check(arg)?;
            }
        }
        E::VectorElement { name, .. } => { binding(name, scopes, locals)?; }
        E::Unmapped(..) => return Err("untyped expression".into()),
        _ => {}
    }
    Ok(())
}

pub(super) fn program(program: &Program) -> Result<(), String> {
    program_mode(program, false)
}
/// Compile-time alternatives have deferred binding joins and are not a native
/// lexical program. Validate operation types and structured delimiters here;
/// the selected program must still pass the full lexical validator above.
pub(super) fn template(program: &Program) -> Result<(), String> {
    program_mode(program, true)
}
fn program_mode(program: &Program, retained: bool) -> Result<(), String> {
    for (launch, sites) in program.launches.iter().enumerate() {
        let locals: HashSet<_> = sites
            .iter()
            .filter_map(|s| declared(&s.statement).map(|(name, _)| name.to_owned()))
            .collect();
        let mut scopes = vec![(Scope::Root, HashMap::new())];
        let mut declarations = HashMap::<String, Option<Type>>::new();
        if retained {
            for site in sites {
                if let Some((name, ty)) = declared(&site.statement) {
                    if let Some(previous) = declarations.insert(name.to_owned(), ty) {
                        if previous != ty { return Err(format!("retained Metal binding `{name}` changes type between alternatives")); }
                    }
                }
            }
        }
        let retained_scope = retained.then(|| [(Scope::Root, declarations)]);
        for (index, site) in sites.iter().enumerate() {
            let context = |reason: String| {
                format!(
                    "compiler error: {reason} in completed Metal launch {launch}, site {index}, source {:?}",
                    site.operation
                )
            };
            if !statement(&site.statement) {
                return Err(context("untyped Metal operation".into()));
            }
            let binding_scopes = retained_scope.as_ref().map_or(scopes.as_slice(), |scope| scope.as_slice());
            let check = |value| typed(value, binding_scopes, &locals).map_err(&context);
            match &site.statement {
                S::Write { name, .. } => { binding(name, binding_scopes, &locals).map_err(&context)?; }
                S::Pointer { base, .. } | S::VectorRead { base, .. } => { binding(base, binding_scopes, &locals).map_err(&context)?; }
                _ => {},
            }
            match &site.statement {
                S::Let { ty, value, .. } | S::Write { ty, value, .. } => {
                    check(value)?;
                    if *ty != value.ty() {
                        return Err(context(
                            "publication requires an explicit conversion".into(),
                        ));
                    }
                }
                S::Assign { name, value } => {
                    check(value)?;
                    let actual = binding_scopes
                        .iter()
                        .rev()
                        .find_map(|(_, names)| names.get(name))
                        .and_then(|t| *t)
                        .ok_or_else(|| {
                            context(format!("assignment to undeclared scalar `{name}`"))
                        })?;
                    if actual != value.ty() {
                        return Err(context(format!("assignment type differs for `{name}`")));
                    }
                }
                S::Evaluate(value) | S::Return(Some(value)) => check(value)?,
                S::If(value) | S::ReturnIf(value) => {
                    check(value)?;
                    if value.ty() != Type::Bool {
                        return Err(context("control predicate must be bool".into()));
                    }
                }
                S::For {
                    start, end, step, ..
                } => {
                    check(start)?;
                    check(end)?;
                    if !integer(start.ty()) || !integer(end.ty()) || *step <= 0 {
                        return Err(context(
                            "loop requires integer bounds and a positive step".into(),
                        ));
                    }
                }
                S::Pointer { index, .. } | S::VectorRead { index, .. } => check(index)?,
                S::MatrixLoad {
                    offset, leading, ..
                }
                | S::MatrixStore {
                    offset, leading, ..
                } => {
                    check(offset)?;
                    check(leading)?;
                }
                _ => {}
            }
            if let S::Write { index, .. } = &site.statement {
                check(index)?;
            }
            match &site.statement {
                S::For { .. } | S::Scope => scopes.push((Scope::Block, HashMap::new())),
                S::If(_) => scopes.push((Scope::Then, HashMap::new())),
                S::Else => {
                    if scopes.last().map(|(kind, _)| *kind) != Some(Scope::Then) {
                        return Err(context("else has no matching if".into()));
                    }
                    scopes.pop();
                    scopes.push((Scope::Else, HashMap::new()));
                }
                S::End => {
                    if scopes.len() == 1 {
                        return Err(context("unmatched scope end".into()));
                    }
                    scopes.pop();
                }
                _ => {}
            }
            if let Some((name, ty)) = declared(&site.statement) {
                if scopes
                    .last_mut()
                    .unwrap()
                    .1
                    .insert(name.into(), ty)
                    .is_some() && !retained
                {
                    return Err(context(format!("duplicate local `{name}` in one scope")));
                }
            }
        }
        if scopes.len() != 1 {
            return Err(format!(
                "compiler error: unclosed scope in completed Metal launch {launch}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::{Site, Type};

    #[test]
    fn rejects_opaque_statements_and_nested_expressions() {
        for statement in [
            S::Unmapped("return;".into()),
            S::Evaluate(E::Builtin(
                "simd_sum".into(),
                vec![E::Unmapped("unrepresented()".into(), Type::F32)],
                Type::F32,
            )),
        ] {
            let program = Program {
                launches: vec![vec![Site {
                    operation: None,
                    statement,
                }]],
            };
            assert!(
                program
                    .validate_typed()
                    .unwrap_err()
                    .contains("compiler error: untyped Metal operation")
            );
        }
    }

    #[test]
    fn rejects_scope_escape_and_inconsistent_publication_types() {
        let local = S::Let {
            name: "x".into(),
            ty: Type::I32,
            value: E::integer(1),
        };
        for (statements, reason) in [
            (vec![S::Scope, local.clone(), S::End, S::Evaluate(E::variable("x", Type::I32))], "outside its lexical scope"),
            (vec![local, S::Assign { name: "x".into(), value: E::Float(1f64.to_bits(), Type::F32) }], "assignment type differs"),
            (vec![S::Else], "else has no matching if"),
        ] {
            let program = Program {
                launches: vec![statements.into_iter().map(|statement| Site { operation: None, statement }).collect()],
            };
            assert!(program.validate_typed().unwrap_err().contains(reason));
        }
    }
}
