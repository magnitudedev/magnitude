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
