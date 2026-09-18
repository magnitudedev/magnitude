//! Lazy traversal of the expansion and producer-materialization space for one specialization.
//! This is coverage of lowering bodies, not of placement, fusion or native schedules.
use super::{lower_selected, Options};
use crate::lowered_ir::{Decision, DecisionRecord, LoweredIr};
use crate::{program::Program, types::Elem};
use std::collections::HashMap;

pub struct Specialization<'a> {
    pub program: &'a Program,
    pub entry: &'a str,
    pub backend: &'a str,
    pub shapes: &'a HashMap<String, i64>,
    pub elements: &'a HashMap<String, Elem>,
    pub options: &'a Options,
}

/// One node of the lowering domain, with no implicit first-choice completion.
/// A completed lowering reports how much of the path it consumed so a compiler
/// can compose its backend execution domains in the same decision tree.
pub enum Expansion {
    Choice(Decision),
    Lowered {
        function: LoweredIr,
        consumed: usize,
    },
}

pub fn expand(request: Specialization<'_>, prefix: &[usize]) -> Result<Expansion, String> {
    let mut consumed = 0;
    let mut pending = None;
    let result = lower_selected(
        request.program,
        request.entry,
        request.backend,
        request.shapes,
        request.elements,
        request.options,
        &mut |domain| {
            let Some(&index) = prefix.get(consumed) else {
                pending = Some(domain.clone());
                return Err("lowering decision is unresolved".into());
            };
            let alternative = domain
                .alternatives
                .get(index)
                .ok_or("lowering choice is outside its derived domain")?;
            consumed += 1;
            Ok(alternative)
        },
    );
    if let Some(domain) = pending {
        return Ok(Expansion::Choice(domain));
    }
    Ok(Expansion::Lowered {
        function: result?,
        consumed,
    })
}

#[derive(Debug)]
pub struct Attempt {
    /// Preorder decisions. Domains include every admitted body, even when a
    /// subsequent expansion fails. Such a failure is not a performance rejection.
    pub steps: Vec<DecisionRecord>,
    pub result: Result<LoweredIr, String>,
}

/// No analysis limit or performance order is implicit here. A caller may stop
/// traversal to enforce its own budget, but then coverage remains incomplete.
/// Failed expansions are returned to the caller, never silently pruned.
pub struct Space<'a> {
    specialization: Specialization<'a>,
    pending: Option<Vec<usize>>,
}

impl<'a> Space<'a> {
    pub fn new(specialization: Specialization<'a>) -> Self {
        Self {
            specialization,
            pending: Some(Vec::new()),
        }
    }

    pub fn exhausted(&self) -> bool {
        self.pending.is_none()
    }
}

impl Iterator for Space<'_> {
    type Item = Attempt;

    fn next(&mut self) -> Option<Self::Item> {
        let prefix = self.pending.take()?;
        let mut steps = Vec::<DecisionRecord>::new();
        let request = &self.specialization;
        let mut result = lower_selected(
            request.program,
            request.entry,
            request.backend,
            request.shapes,
            request.elements,
            request.options,
            &mut |domain| {
                let index = prefix.get(steps.len()).copied().unwrap_or(0);
                let selected = domain.alternatives.get(index)
                    .ok_or("lowering replay index is outside its derived domain")?;
                steps.push(DecisionRecord {
                    domain: domain.clone(),
                    selected: selected.clone(),
                });
                Ok(selected)
            },
        );
        if result.is_ok() && steps.len() < prefix.len() {
            result = Err("lowering replay ended before consuming its decision prefix".into());
        }
        // Advance the lexicographic DFS cursor without allocating every sibling.
        // A billion-capacity interval needs only one index per decision depth.
        for depth in (0..steps.len()).rev() {
            let index = prefix.get(depth).copied().unwrap_or(0);
            if index + 1 < steps[depth].domain.alternatives.len() {
                let mut next = (0..=depth).map(|i| prefix.get(i).copied().unwrap_or(0)).collect::<Vec<_>>();
                next[depth] = index + 1;
                self.pending = Some(next);
                break;
            }
        }
        Some(Attempt { steps, result })
    }
}

/// Replay a saved path against freshly derived domains. This validates decision
/// applicability and consumption, not program identity or performance optimality.
pub fn replay(
    request: Specialization<'_>,
    expected: &[DecisionRecord],
) -> Result<LoweredIr, String> {
    let mut cursor = 0;
    let result = lower_selected(
        request.program,
        request.entry,
        request.backend,
        request.shapes,
        request.elements,
        request.options,
        &mut |domain| {
            let record = expected
                .get(cursor)
                .ok_or("decision replay is incomplete")?;
            if record.domain != *domain {
                return Err(format!("decision domain changed at step {cursor}"));
            }
            cursor += 1;
            Ok(record.selected.clone())
        },
    )?;
    if cursor != expected.len() {
        return Err("decision replay contains unused decisions".into());
    }
    Ok(result)
}
