//! Contiguous snapshot transfers selected on the retained typed implementation.
//! Packed MSL vectors retain scalar alignment and exact 2/3/4-element size.
//! This is a source vector access contract, not a promised native instruction.
use super::{Expression as E, Program, Site, Space, Statement as S, Type as T};
use seismic_lang::{exec::ir::OperationId, sym::{Atom, Sym}, syntax::ast::BinaryOp as B};
use std::collections::{HashMap, HashSet};
use super::rewrite::substitute;
mod bundle;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind { CopyLoop, ReadBundle }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice { pub kind: Kind, pub launch: usize, pub site: usize, pub operation: Option<OperationId>, pub maximum: u8 }
impl Choice {
    pub fn len(&self) -> usize { usize::from(self.maximum) }
    pub fn is_empty(&self) -> bool { self.maximum == 0 }
    pub fn get(&self, index: usize) -> Option<u8> { (index < self.len()).then_some(index as u8 + 1) }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection { pub choice: Choice, pub width: u8 }

fn binding_names(body: &[Site]) -> HashSet<String> {
    body.iter().filter_map(|site| match &site.statement {
        S::For { name, .. } | S::Let { name, .. } | S::Assign { name, .. }
        | S::Array { name, .. } | S::Pointer { name, .. } | S::Fragment { name, .. }
        | S::VectorRead { name, .. } => Some(name.clone()),
        _ => None,
    }).collect()
}

struct CopyLoop {
    close: usize, name: String, first: i64, end: i64,
    destination: String, destination_index: E, destination_type: T,
    source: String, index: E, ty: T, count: Option<E>,
    value: E, guards: Vec<E>,
}
fn affine(e: &E) -> Option<Sym> {
    Some(match e {
        E::Integer(n, _) => Sym::constant(*n),
        E::Variable(name, _) | E::Parameter { name, .. } => Sym::param(name),
        E::Cast(_, a) | E::Bitcast(_, a) => affine(a)?,
        E::Binary(B::Add, a, b, _) => affine(a)?.add(&affine(b)?),
        E::Binary(B::Sub, a, b, _) => affine(a)?.sub(&affine(b)?),
        E::Binary(B::Mul, a, b, _) => affine(a)?.mul(&affine(b)?),
        _ => return None,
    })
}
fn read(e: &E) -> Option<(String, E, T, Option<E>)> {
    match e {
        E::Cast(_, value) | E::Bitcast(_, value) => read(value),
        E::Read { name, index, space: Space::Device, ty } => Some((name.clone(), *index.clone(), *ty, None)),
        E::Helper(crate::support::Helper::Read, args, ty) => {
            let [E::Variable(name, T::U64), index, count] = args.as_slice() else { return None; };
            Some((name.clone(), index.clone(), *ty, Some(count.clone())))
        }
        _ => None,
    }
}
fn copy_facts(body: &[Site]) -> HashMap<usize, super::simplify::Facts> {
    // Retain lexical environments only at possible copy sites. A large terminal
    // body must not acquire a full scalar environment for every instruction.
    let mut retained = HashSet::new();
    for (at, site) in body.iter().enumerate() {
        if !matches!(&site.statement, S::For { start: E::Integer(_, T::I32), end: E::Integer(_, T::I32), step: 1, .. }) { continue; }
        let mut position = at + 1;
        while body.get(position).is_some_and(|s| matches!(s.statement, S::Let { .. } | S::If(_))) { position += 1; }
        if body.get(position).is_some_and(|s| matches!(s.statement, S::Write { space: Space::Private, .. })) {
            retained.extend(at + 1..=position);
        }
    }
    super::simplify::scope_facts(body, &retained)
}
fn recognized(body: &[Site], at: usize, facts: &HashMap<usize, super::simplify::Facts>) -> Option<CopyLoop> {
    let S::For { name, start: E::Integer(first, T::I32), end: E::Integer(end, T::I32), step: 1 } = &body[at].statement else { return None; };
    if *first < 0 || end - first < 2 || *end > i64::from(i32::MAX) { return None; }
    let mut position = at + 1;
    let mut guards = Vec::new();
    let mut bindings: Vec<(String, E)> = Vec::new();
    let inline = |e: &E, bindings: &[(String, E)]| {
        bindings.iter().rev().fold(e.clone(), |e, (name, value)| substitute(&e, name, value, None))
    };
    loop {
        match &body.get(position)?.statement {
            S::Let { name, ty, value } => {
                super::simplify::bounds_in(value, facts.get(&position)?)?;
                bindings.push((name.clone(), value.clone().cast(*ty))); position += 1;
            }
            S::If(condition) => {
                super::simplify::bounds_in(condition, facts.get(&position)?)?;
                guards.push(inline(condition, &bindings)); position += 1;
            }
            _ => break,
        }
    }
    let S::Write { name: destination, index: destination_index, space: Space::Private, ty: destination_type, value } = &body.get(position)?.statement else { return None; };
    let value = inline(value, &bindings);
    let destination_index = inline(destination_index, &bindings);
    let (source, index, ty, count) = read(&value)?;
    if matches!(ty, T::Bool | T::U64) { return None; } // No packed bool/ulong source cover in this family.
    let (lo, hi) = super::simplify::bounds_in(&index, facts.get(&position)?)?;
    if lo < 0 || hi > i128::from(i64::MAX) { return None; }
    super::simplify::bounds_in(&destination_index, facts.get(&position)?)?;
    if let Some(count) = &count {
        super::simplify::bounds_in(count, facts.get(&position)?)?;
        if affine(count)?.params().contains(name) { return None; }
    }
    if affine(&index)?.linear_in(&Atom::Param(name.clone()))?.0 != 1 { return None; }
    let close = position + guards.len() + 1;
    if !(position + 1..=close).all(|i| body.get(i).is_some_and(|s| s.statement == S::End)) { return None; }
    Some(CopyLoop { close, name: name.clone(), first: *first, end: *end, destination: destination.clone(), destination_index: destination_index.clone(), destination_type: *destination_type, source, index, ty, count, value: value.clone(), guards })
}
pub fn choices(program: &Program) -> Vec<Choice> {
    let mut result = Vec::new();
    for (launch, body) in program.launches().iter().enumerate() {
        if body.iter().any(|s| matches!(s.statement, S::Unmapped(_))) { continue; }
        let facts = copy_facts(body);
        for (site, statement) in body.iter().enumerate() {
            if let Some(copy) = recognized(body, site, &facts) {
                result.push(Choice { kind: Kind::CopyLoop, launch, site, operation: statement.operation, maximum: (copy.end - copy.first).min(4) as u8 });
            }
        }
        result.extend(bundle::recognize(body).into_iter().map(|bundle| {
            let site = bundle.sites[0];
            Choice { kind: Kind::ReadBundle, launch, site, operation: body[site].operation, maximum: bundle.sites.len().min(4) as u8 }
        }));
    }
    result.sort_by_key(|choice| (choice.launch, choice.site));
    result
}


pub(crate) fn apply(body: &mut Vec<Site>, launch: usize, selections: &[Selection]) -> Result<(), String> {
    let facts = copy_facts(body);
    let mut selected = HashMap::new();
    for selection in selections.iter().filter(|s| s.choice.launch == launch && s.choice.kind == Kind::CopyLoop) {
        let at = selection.choice.site;
        let copy = recognized(body, at, &facts).ok_or("selected transfer no longer matches its retained copy")?;
        let maximum = (copy.end - copy.first).min(4) as u8;
        if selection.choice.operation != body[at].operation || selection.choice.maximum != maximum || selection.width == 0 || selection.width > maximum || selected.insert(at, (selection.width, copy)).is_some() { return Err("invalid terminal transfer selection".into()); }
    }
    let mut names = binding_names(body);
    let bundles: HashMap<_, _> = bundle::recognize(body).into_iter().map(|bundle| (bundle.sites[0], bundle)).collect();
    let mut replacements = HashMap::new();
    let mut selected_bundles = HashSet::new();
    for selection in selections.iter().filter(|s| s.choice.launch == launch && s.choice.kind == Kind::ReadBundle) {
        let at = selection.choice.site;
        let bundle = bundles.get(&at).ok_or("selected transfer no longer matches its retained read bundle")?;
        let maximum = bundle.sites.len().min(4) as u8;
        if selection.choice.operation != body[at].operation || selection.choice.maximum != maximum || selection.width == 0 || selection.width > maximum || !selected_bundles.insert(at) { return Err("invalid terminal read bundle selection".into()); }
        replacements.extend(bundle::replacements(body, bundle, selection.width, launch, &mut names));
    }
    let mut output = Vec::new();
    let mut at = 0;
    while at < body.len() {
        let Some((width, copy)) = selected.get(&at).filter(|(width, _)| *width > 1) else {
            if let Some(replacement) = replacements.remove(&at) { output.extend(replacement); }
            else { output.push(body[at].clone()); }
            at += 1; continue;
        };
        output.extend(copy_replacement(body, at, copy, *width, launch, &mut names));
        at = copy.close + 1;
    }
    *body = output;
    Ok(())
}

/// A replacement owns exactly the recognized source interval. This is shared
/// by unresolved family construction and explicit selected reconstruction.
fn copy_replacement(body: &[Site], at: usize, copy: &CopyLoop, width: u8,
    launch: usize, names: &mut HashSet<String>) -> Vec<Site> {
    if width == 1 { return body[at..=copy.close].to_vec(); }
    let mut output = Vec::new();
    let site = |statement| Site { operation: body[at].operation, statement };
    let width = i64::from(width);
    let complete = copy.first + (copy.end - copy.first) / width * width;
    let mut vector = format!("seismic_transfer_{launch}_{at}");
    while !names.insert(vector.clone()) { vector.push('_'); }
    let coordinate = |offset| E::binary(B::Add, E::variable(&copy.name, T::I32), E::integer(offset), T::I32);
    let and = |left, right| E::ShortCircuit { or: false, left: Box::new(left), right: Box::new(right) };
    let mut guard = E::Integer(1, T::Bool);
    for offset in 0..width {
        for condition in &copy.guards { guard = and(guard, substitute(condition, &copy.name, &coordinate(offset), None)); }
    }
    if let Some(count) = &copy.count {
        // The original helper remains in the scalar fallback. Fast-path
        // bounds use widened arithmetic and cannot overflow on count-width.
        let count = count.clone().cast(T::I64);
        let first = copy.index.clone().cast(T::I64);
        let valid = and(E::binary(B::Ge, first.clone(), E::Integer(0, T::I64), T::Bool),
            and(E::binary(B::Ge, count.clone(), E::Integer(width, T::I64), T::Bool), E::binary(B::Le, first, E::binary(B::Sub, count, E::Integer(width, T::I64), T::I64), T::Bool)));
        guard = and(guard, valid);
    }
    output.push(site(S::For { name: copy.name.clone(), start: E::integer(copy.first), end: E::integer(complete), step: width }));
    output.push(site(S::If(guard)));
    output.push(site(S::VectorRead { name: vector.clone(), base: copy.source.clone(), index: copy.index.clone(), ty: copy.ty, components: width as u8 }));
    for offset in 0..width {
        output.push(site(S::Write { name: copy.destination.clone(), index: substitute(&copy.destination_index, &copy.name, &coordinate(offset), None), space: Space::Private, ty: copy.destination_type,
            value: substitute(&copy.value, &copy.name, &coordinate(offset), Some(&E::VectorElement { name: vector.clone(), component: offset as u8, ty: copy.ty })) }));
    }
    output.push(site(S::Else));
    for offset in 0..width {
        for condition in &copy.guards { output.push(site(S::If(substitute(condition, &copy.name, &coordinate(offset), None)))); }
        output.push(site(S::Write { name: copy.destination.clone(), index: substitute(&copy.destination_index, &copy.name, &coordinate(offset), None), space: Space::Private, ty: copy.destination_type, value: substitute(&copy.value, &copy.name, &coordinate(offset), None) }));
        for _ in &copy.guards { output.push(site(S::End)); }
    }
    output.push(site(S::End)); output.push(site(S::End));
    if complete < copy.end {
        output.push(site(S::For { name: copy.name.clone(), start: E::integer(complete), end: E::integer(copy.end), step: 1 }));
        output.extend_from_slice(&body[at + 1..=copy.close]);
    }
    output
}
