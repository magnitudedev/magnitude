//! Adjacent device loads with only total scalar definitions between them.
//! Recognition uses the common integer analysis; no decoder or source names
//! participate. Effectful checked loads stay scalar until their checks discharge.
use super::{E, S, T, Site, Sym, HashMap, HashSet};
use crate::terminal::simplify;

pub(super) struct Bundle { pub sites: Vec<usize> }
struct Load { site: usize, base: String, ty: T, index: Sym }

pub(super) fn recognize(body: &[Site]) -> Vec<Bundle> {
    let writes: HashSet<_> = body.iter().filter_map(|site| match &site.statement {
        S::Assign { name, .. } => Some(name.clone()), _ => None,
    }).collect();
    let mut aliases = HashMap::<String, Sym>::new();
    let mut scopes = Vec::new();
    let mut result = Vec::new();
    let mut run: Vec<Load> = Vec::new();
    let finish = |run: &mut Vec<Load>, result: &mut Vec<Bundle>| {
        if run.len() >= 2 { result.push(Bundle { sites: run.iter().map(|load| load.site).collect() }); }
        run.clear();
    };
    simplify::walk_facts(&mut body.to_vec(), |site, statement, facts| {
        let symbol = |e: &E| simplify::symbolic_with(e, facts, &|name| aliases.get(name).cloned());
        let mut transparent = false;
        if let S::Let { name, value, ty } = &statement.statement {
            let scalar = symbol(&value.clone().cast(*ty));
            if let Some((base, index, read_type, None)) = super::read(value) {
                if !matches!(read_type, T::Bool | T::U64) {
                    if let Some(index) = symbol(&index) {
                        if let Some(first) = run.first() {
                            if first.base != base || first.ty != read_type || index.sub(&first.index).as_constant() != Some(run.len() as i64) {
                                finish(&mut run, &mut result);
                            }
                        }
                        run.push(Load { site, base, ty: read_type, index });
                        transparent = true;
                    }
                }
            } else if simplify::bounds_in(&value.clone().cast(*ty), facts).is_some() {
                transparent = true;
            }
            // A captured mutable value is a fresh identity, not its value at
            // a later read. Scope identities also prevent accidental equality
            // between shadowed induction variables or scalar definitions.
            let captured = scalar.filter(|s| !s.params().iter().any(|name| writes.contains(name)));
            aliases.insert(name.clone(), captured.unwrap_or_else(|| Sym::param(&format!("$transfer_value_{site}"))));
            if writes.contains(name) { aliases.remove(name); }
        }
        if !transparent { finish(&mut run, &mut result); }
        match &statement.statement {
            S::Scope | S::If(_) => scopes.push(aliases.clone()),
            S::For { name, .. } => {
                scopes.push(aliases.clone());
                if !writes.contains(name) { aliases.insert(name.clone(), Sym::param(&format!("$transfer_induction_{site}"))); }
                else { aliases.remove(name); }
            }
            S::Else => { if let Some(parent) = scopes.last() { aliases = parent.clone(); } }
            S::End => { if let Some(parent) = scopes.pop() { aliases = parent; } }
            _ => {}
        }
    });
    finish(&mut run, &mut result);
    result
}

/// Insert each packed read at the first scalar load that it replaces. Keep
/// scalar bindings and intervening definitions in their original scopes/order.
pub(super) fn replacements(body: &[Site], bundle: &Bundle, width: u8, launch: usize, names: &mut HashSet<String>) -> HashMap<usize, Vec<Site>> {
    let mut result = HashMap::new();
    if width == 1 { return result; }
    for chunk in bundle.sites.chunks_exact(usize::from(width)) {
        let at = chunk[0];
        let S::Let { value, .. } = &body[at].statement else { unreachable!() };
        let (base, index, element, None) = super::read(value).unwrap() else { unreachable!() };
        let mut vector = format!("seismic_transfer_{launch}_{at}");
        while !names.insert(vector.clone()) { vector.push('_'); }
        for (component, &at) in chunk.iter().enumerate() {
            let S::Let { name, value, ty } = &body[at].statement else { unreachable!() };
            let site = |statement| Site { operation: body[at].operation, statement };
            let mut replacement = Vec::new();
            if component == 0 { replacement.push(site(S::VectorRead { name: vector.clone(), base: base.clone(), index: index.clone(), ty: element, components: width })); }
            replacement.push(site(S::Let { name: name.clone(), ty: *ty, value: super::substitute(value, "", &E::integer(0), Some(&E::VectorElement { name: vector.clone(), component: component as u8, ty: element })) }));
            result.insert(at, replacement);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::syntax::ast::BinaryOp as B;
    use crate::terminal::{Program, Space, transfer::{self, Kind, Selection}};
    fn site(statement: S) -> Site { Site { operation: None, statement } }
    fn load(name: &str, index: E) -> S {
        S::Let { name: name.into(), ty: T::U32, value: E::Read { name: "words".into(), index: Box::new(index), space: Space::Device, ty: T::U32 } }
    }
    fn index(offset: i64) -> E { E::binary(B::Add, E::variable("base", T::I32), E::integer(offset), T::I32).cast(T::I64) }
    fn program() -> Program {
        let mut body = vec![site(S::For { name: "segment".into(), start: E::integer(0), end: E::integer(256), step: 1 }),
            site(S::Let { name: "base".into(), ty: T::I32, value: E::binary(B::Mul, E::variable("segment", T::I32), E::integer(8), T::I32) })];
        for offset in 0..7 {
            body.push(site(S::Let { name: format!("index{offset}"), ty: T::I64, value: index(offset) }));
            body.push(site(load(&format!("word{offset}"), E::variable(format!("index{offset}"), T::I64))));
        }
        body.push(site(S::End));
        Program { launches: vec![body] }
    }
    #[test]
    fn adjacent_loads_share_a_complete_width_domain_and_keep_scalar_tails() {
        let program = program();
        let choices = transfer::choices(&program);
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].kind, Kind::ReadBundle);
        assert_eq!(choices[0].maximum, 4);
        for width in 1..=4 {
            let mut body = program.launches[0].clone();
            transfer::apply(&mut body, 0, &[Selection { choice: choices[0].clone(), width }]).unwrap();
            let vectors = body.iter().filter(|s| matches!(s.statement, S::VectorRead { .. })).count();
            let scalars = body.iter().filter(|s| matches!(s.statement, S::Let { value: E::Read { .. }, .. })).count();
            assert_eq!(vectors, if width == 1 { 0 } else { 7 / usize::from(width) });
            assert_eq!(scalars, if width == 1 { 7 } else { 7 % usize::from(width) });
            assert_eq!(body.iter().filter(|s| matches!(s.statement, S::Let { .. })).count(), 15);
        }
    }
    #[test]
    fn writes_checks_and_mutable_captures_do_not_become_load_bundles() {
        let original = program();
        for barrier in [S::FailureStatus, S::Write { name: "words".into(), index: E::Integer(0, T::I64), space: Space::Device, ty: T::U32, value: E::Integer(3, T::U32) },
            S::Evaluate(E::Helper(crate::support::Helper::Index, vec![E::variable("unknown", T::I64), E::Integer(4, T::I64)], T::I64))] {
            let mut body = original.launches[0][..6].to_vec(); // two loads
            body.insert(4, site(barrier));
            body.push(site(S::End));
            assert!(recognize(&body).is_empty());
        }
        let body = vec![site(S::Let { name: "cursor".into(), ty: T::I32, value: E::integer(0) }),
            site(S::Let { name: "saved".into(), ty: T::I32, value: E::variable("cursor", T::I32) }),
            site(S::Assign { name: "cursor".into(), value: E::integer(9) }),
            site(load("a", E::variable("saved", T::I32))),
            site(load("b", E::binary(B::Add, E::variable("cursor", T::I32), E::integer(1), T::I32)))];
        assert!(recognize(&body).is_empty(), "saved cursor cannot be replaced by its later value");
    }
}
