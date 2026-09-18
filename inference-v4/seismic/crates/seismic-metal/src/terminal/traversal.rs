//! Typed traversal of constant, finite terminal loops. It changes execution
//! instructions before both emission and accounting, without native feedback.
use super::{Expression as E, Program, Site, Statement as S, Type as T};
use seismic_lang::{ast::BinaryOp as B, ir::OperationId};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub launch: usize,
    pub site: usize,
    pub operation: Option<OperationId>,
    pub iterations: usize,
}
impl Choice {
    pub fn len(&self) -> usize {
        self.iterations
    }
    pub fn is_empty(&self) -> bool {
        self.iterations == 0
    }
    pub fn get(&self, index: usize) -> Option<usize> {
        (index < self.iterations).then_some(index + 1)
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub choice: Choice,
    pub width: usize,
}

pub fn choices(program: &Program) -> Result<Vec<Choice>, String> {
    let mut choices = Vec::new();
    for (launch, body) in program.launches().iter().enumerate() {
        // Text has no admitted scope/effect model. It cannot participate in a
        // typed loop rewrite even if neighboring statements are understood.
        if body.iter().any(|s| matches!(s.statement, S::Unmapped(_))) {
            continue;
        }
        for (site, statement) in body.iter().enumerate() {
            if let Some(iterations) = iterations(body, site)? {
                if iterations > 1 {
                    choices.push(Choice {
                        launch,
                        site,
                        operation: statement.operation,
                        iterations,
                    });
                }
            }
        }
    }
    Ok(choices)
}
fn iterations(body: &[Site], at: usize) -> Result<Option<usize>, String> {
    let S::For {
        name,
        start: E::Integer(start, T::I32),
        end: E::Integer(end, T::I32),
        step,
    } = &body[at].statement
    else {
        return Ok(None);
    };
    if *step <= 0 || *start >= *end {
        return Ok(None);
    }
    let count = (i128::from(*end) - i128::from(*start) + i128::from(*step) - 1) / i128::from(*step);
    let after = i128::from(*start) + count * i128::from(*step);
    if *start < i64::from(i32::MIN) || *end > i64::from(i32::MAX) || after > i128::from(i32::MAX) {
        return Ok(None);
    }
    let end = close(body, at)?;
    if body[at + 1..end]
        .iter()
        .any(|s| matches!(&s.statement, S::Assign { name: target, .. } if target == name))
    {
        return Ok(None);
    }
    Ok(usize::try_from(count).ok())
}
fn close(body: &[Site], at: usize) -> Result<usize, String> {
    let mut depth = 0;
    for (index, site) in body.iter().enumerate().skip(at + 1) {
        match site.statement {
            S::For { .. } | S::If(_) | S::Scope => depth += 1,
            S::End if depth == 0 => return Ok(index),
            S::End => depth -= 1,
            _ => {}
        }
    }
    Err("unclosed terminal traversal scope".into())
}

/// Original site indices name choices even when an enclosing body is copied.
/// Nested choices are applied once to the retained body, then copied intact.
pub(crate) fn apply(
    body: &mut Vec<Site>,
    launch: usize,
    selections: &[Selection],
) -> Result<(), String> {
    let mut widths = std::collections::HashMap::new();
    for selected in selections.iter().filter(|s| s.choice.launch == launch) {
        let choice = &selected.choice;
        if selected.width == 0
            || selected.width > choice.iterations
            || body
                .get(choice.site)
                .is_none_or(|s| s.operation != choice.operation)
            || iterations(body, choice.site)? != Some(choice.iterations)
            || widths.insert(choice.site, selected.width).is_some()
        {
            return Err("terminal traversal selection does not match its retained loop".into());
        }
    }
    if selections
        .iter()
        .all(|s| s.choice.launch != launch || s.width == 1)
    {
        return Ok(());
    }
    let mut names: HashSet<_> = body
        .iter()
        .filter_map(|s| match &s.statement {
            S::For { name, .. }
            | S::Let { name, .. }
            | S::Assign { name, .. }
            | S::Array { name, .. }
            | S::Pointer { name, .. }
            | S::Fragment { name, .. } | S::VectorRead { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    fn expand(
        body: &[Site],
        start: usize,
        end: usize,
        launch: usize,
        widths: &std::collections::HashMap<usize, usize>,
        names: &mut HashSet<String>,
    ) -> Result<Vec<Site>, String> {
        let mut output = Vec::new();
        let mut at = start;
        while at < end {
            let site = &body[at];
            if let S::For {
                name,
                start,
                end: _,
                step,
            } = &site.statement
            {
                let close_at = close(body, at)?;
                let inner = expand(body, at + 1, close_at, launch, widths, names)?;
                let width = widths.get(&at).copied().unwrap_or(1);
                if width > 1 {
                    let E::Integer(first, T::I32) = start else {
                        return Err("unroll needs a constant loop origin".into());
                    };
                    let count =
                        iterations(body, at)?.ok_or("unroll needs a finite constant loop")?;
                    let complete = count / width;
                    let mut chunk = format!("seismic_traversal_{launch}_{at}");
                    while !names.insert(chunk.clone()) {
                        chunk.push('_');
                    }
                    let statement = |statement| Site {
                        operation: site.operation,
                        statement,
                    };
                    let occurrence = |index: E, destination: &mut Vec<Site>| {
                        destination.push(statement(S::Scope));
                        destination.push(statement(S::Let {
                            name: name.clone(),
                            ty: T::I32,
                            value: index,
                        }));
                        destination.extend(inner.iter().cloned());
                        destination.push(statement(S::End));
                    };
                    if complete > 1 {
                        output.push(statement(S::For {
                            name: chunk.clone(),
                            start: E::integer(0),
                            end: E::integer(complete as i64),
                            step: 1,
                        }));
                    }
                    for offset in 0..width {
                        let ordinal = if complete == 1 {
                            E::Integer(offset as i64, T::I64)
                        } else {
                            E::binary(
                                B::Add,
                                E::binary(
                                    B::Mul,
                                    E::variable(&chunk, T::I32).cast(T::I64),
                                    E::Integer(width as i64, T::I64),
                                    T::I64,
                                ),
                                E::Integer(offset as i64, T::I64),
                                T::I64,
                            )
                        };
                        let value = E::binary(
                            B::Add,
                            E::Integer(*first, T::I64),
                            E::binary(B::Mul, ordinal, E::Integer(*step, T::I64), T::I64),
                            T::I64,
                        )
                        .cast(T::I32);
                        occurrence(value, &mut output);
                    }
                    if complete > 1 {
                        output.push(statement(S::End));
                    }
                    for offset in complete * width..count {
                        occurrence(E::integer(*first + offset as i64 * *step), &mut output);
                    }
                } else {
                    output.push(site.clone());
                    output.extend(inner);
                    output.push(body[close_at].clone());
                }
                at = close_at + 1;
            } else {
                output.push(site.clone());
                at += 1;
            }
        }
        Ok(output)
    }
    *body = expand(body, 0, body.len(), launch, &widths, &mut names)?;
    Ok(())
}

/// Operation classes preserved occurrence-for-occurrence by this traversal and
/// its integer-only realization. Control, address arithmetic, integer casts and helper
/// checks can disappear; their counts are deliberately not used for a region.
pub(crate) fn preserves(primitive: &super::Primitive) -> bool {
    use super::{Primitive as P, Space};
    let floating = |ty: &T| matches!(ty, T::F16 | T::BF16 | T::F32);
    match primitive {
        P::VectorRead { .. } => true,
        P::Read { space, .. } => *space != Space::Constant,
        P::Write { .. } | P::Barrier | P::MatrixLoad { .. }
        | P::MatrixStore { .. } | P::MatrixMultiplyAccumulate { .. } => true,
        P::Binary { ty, .. } | P::Unary { ty, .. } => floating(ty),
        // Explicit floating publication boundaries survive integer unrolling
        // and regrouping, including BF16's required narrowing/widening pairs.
        P::Cast { from, to } => floating(from) && floating(to),
        P::Builtin { result, .. } => floating(result),
        _ => false,
    }
}
