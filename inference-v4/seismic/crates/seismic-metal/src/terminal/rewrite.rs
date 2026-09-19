//! Structural substitution within typed terminal expressions.
use super::Expression as E;

pub(super) fn substitute(e: &E, name: &str, value: &E, replacement: Option<&E>) -> E {
    let recurse = |e: &E| substitute(e, name, value, replacement);
    match e {
        E::Variable(n, _) if n == name => value.clone(),
        E::Read { .. } | E::Helper(crate::support::Helper::Read, ..) if replacement.is_some() => replacement.unwrap().clone(),
        E::Binary(op, a, b, ty) => E::Binary(*op, Box::new(recurse(a)), Box::new(recurse(b)), *ty),
        E::Unary(op, a, ty) => E::Unary(*op, Box::new(recurse(a)), *ty),
        E::Cast(ty, a) => E::Cast(*ty, Box::new(recurse(a))),
        E::Bitcast(ty, a) => E::Bitcast(*ty, Box::new(recurse(a))),
        E::Select(c, a, b) => E::Select(Box::new(recurse(c)), Box::new(recurse(a)), Box::new(recurse(b))),
        E::EagerSelect(c, a, b) => E::EagerSelect(Box::new(recurse(c)), Box::new(recurse(a)), Box::new(recurse(b))),
        E::ShortCircuit { or, left, right } => E::ShortCircuit { or: *or, left: Box::new(recurse(left)), right: Box::new(recurse(right)) },
        E::Builtin(n, args, ty) => E::Builtin(n.clone(), args.iter().map(recurse).collect(), *ty),
        E::Helper(h, args, ty) => E::Helper(*h, args.iter().map(recurse).collect(), *ty),
        E::Read { name, index, space, ty } => E::Read { name: name.clone(), index: Box::new(recurse(index)), space: *space, ty: *ty },
        _ => e.clone(),
    }
}

/// Substitute a retained compile-time operand in the typed operation, keeping
/// the template's operation graph intact after specialization.
pub(crate) fn statement(statement: &super::Statement, name: &str, value: &E) -> super::Statement {
    use super::Statement as S;
    let expr = |e: &E| substitute(e, name, value, None);
    match statement {
        S::Let { name, ty, value } => S::Let { name: name.clone(), ty: *ty, value: expr(value) },
        S::Assign { name, value } => S::Assign { name: name.clone(), value: expr(value) },
        S::Pointer { name, base, index, space, ty } => S::Pointer { name: name.clone(), base: base.clone(), index: expr(index), space: *space, ty: *ty },
        S::VectorRead { name, base, index, ty, components } => S::VectorRead { name: name.clone(), base: base.clone(), index: expr(index), ty: *ty, components: *components },
        S::Write { name, index, space, ty, value } => S::Write { name: name.clone(), index: expr(index), space: *space, ty: *ty, value: expr(value) },
        S::Evaluate(value) => S::Evaluate(expr(value)),
        S::For { name, start, end, step } => S::For { name: name.clone(), start: expr(start), end: expr(end), step: *step },
        S::If(value) => S::If(expr(value)),
        S::ReturnIf(value) => S::ReturnIf(expr(value)),
        S::Return(value) => S::Return(value.as_ref().map(expr)),
        S::MatrixLoad { fragment, layout, base, offset, leading, space, transpose } => S::MatrixLoad { fragment: fragment.clone(), layout: layout.clone(), base: base.clone(), offset: expr(offset), leading: expr(leading), space: *space, transpose: *transpose },
        S::MatrixStore { fragment, layout, base, offset, leading, space } => S::MatrixStore { fragment: fragment.clone(), layout: layout.clone(), base: base.clone(), offset: expr(offset), leading: expr(leading), space: *space },
        S::Array { .. } | S::Else | S::Scope | S::End | S::FailureStatus | S::Barrier | S::Fragment { .. } | S::MatrixMultiplyAccumulate { .. } | S::Unmapped(_) => statement.clone(),
    }
}
