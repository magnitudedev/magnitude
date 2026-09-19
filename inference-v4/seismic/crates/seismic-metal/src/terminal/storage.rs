//! Local storage element types. A thread or threadgroup array never holds half-width
//! floats natively (see `seismic_realization::dispatch::local_storage_dtype`): its
//! declaration is already widened, and this pass makes every access agree with it. A read
//! narrows back to the logical type (exact: the stored value is representable); a write
//! rounds to the logical type first and then widens, so assignment rounding is unchanged.
use super::{Expression as E, Site, Space, Statement as S, Type as T};
use std::collections::HashMap;

fn local(space: Space) -> bool {
    matches!(space, Space::Private | Space::Threadgroup)
}

fn reads(e: E, stored: &HashMap<String, T>) -> E {
    let r = |e: Box<E>| Box::new(reads(*e, stored));
    match e {
        E::Read { name, index, space, ty } => {
            let index = r(index);
            match stored.get(&name).copied().filter(|held| local(space) && *held != ty) {
                Some(held) => E::Cast(ty, Box::new(E::Read { name, index, space, ty: held })),
                None => E::Read { name, index, space, ty },
            }
        }
        E::Binary(op, a, b, ty) => E::Binary(op, r(a), r(b), ty),
        E::Unary(op, a, ty) => E::Unary(op, r(a), ty),
        E::Cast(ty, a) => E::Cast(ty, r(a)),
        E::Bitcast(ty, a) => E::Bitcast(ty, r(a)),
        E::Select(c, a, b) => E::Select(r(c), r(a), r(b)),
        E::EagerSelect(c, a, b) => E::EagerSelect(r(c), r(a), r(b)),
        E::ShortCircuit { or, left, right } => E::ShortCircuit { or, left: r(left), right: r(right) },
        E::Builtin(name, args, ty) => E::Builtin(name, args.into_iter().map(|a| reads(a, stored)).collect(), ty),
        E::Helper(helper, args, ty) => E::Helper(helper, args.into_iter().map(|a| reads(a, stored)).collect(), ty),
        other => other,
    }
}

/// Idempotent: an access already typed as its storage is left alone.
pub(super) fn widen(sites: &mut [Site]) {
    let mut stored: HashMap<String, T> = HashMap::new();
    for site in sites.iter_mut() {
        let statement = std::mem::replace(&mut site.statement, S::End);
        let e = |e: E| reads(e, &stored);
        site.statement = match statement {
            S::Array { name, ty, elements } => {
                stored.insert(name.clone(), ty);
                S::Array { name, ty, elements }
            }
            S::Pointer { name, base, index, space, ty } => {
                let index = e(index);
                // A pointer into local storage has its base's element type.
                let ty = stored.get(&base).copied().filter(|_| local(space)).unwrap_or(ty);
                if local(space) {
                    stored.insert(name.clone(), ty);
                }
                S::Pointer { name, base, index, space, ty }
            }
            S::Write { name, index, space, ty, value } => {
                let (index, value) = (e(index), e(value));
                match stored.get(&name).copied().filter(|held| local(space) && *held != ty) {
                    Some(held) => S::Write { name, index, space, ty: held, value: value.cast(ty).cast(held) },
                    None => S::Write { name, index, space, ty, value },
                }
            }
            S::Let { name, ty, value } => S::Let { name, ty, value: e(value) },
            S::Assign { name, value } => S::Assign { name, value: e(value) },
            S::VectorRead { name, base, index, ty, components } => S::VectorRead { name, base, index: e(index), ty, components },
            S::Evaluate(value) => S::Evaluate(e(value)),
            S::For { name, start, end, step } => S::For { name, start: e(start), end: e(end), step },
            S::If(value) => S::If(e(value)),
            S::ReturnIf(value) => S::ReturnIf(e(value)),
            S::Return(value) => S::Return(value.map(e)),
            S::MatrixLoad { fragment, layout, base, offset, leading, space, transpose } => {
                S::MatrixLoad { fragment, layout, base, offset: e(offset), leading: e(leading), space, transpose }
            }
            S::MatrixStore { fragment, layout, base, offset, leading, space } => S::MatrixStore { fragment, layout, base, offset: e(offset), leading: e(leading), space },
            other => other,
        };
    }
}
