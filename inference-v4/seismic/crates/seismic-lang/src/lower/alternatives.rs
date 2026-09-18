//! Lazy traversal of the expansion and producer-materialization space for one specialization.
//! This is coverage of lowering bodies, not of placement, fusion or native schedules.
use super::{lower_selected, Options};
use crate::lowered_ir::{Alternative, DecisionRecord, LoweredIr};
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
    pending: Vec<Vec<Alternative>>,
}

impl<'a> Space<'a> {
    pub fn new(specialization: Specialization<'a>) -> Self {
        Self {
            specialization,
            pending: vec![Vec::new()],
        }
    }

    pub fn exhausted(&self) -> bool {
        self.pending.is_empty()
    }
}

impl Iterator for Space<'_> {
    type Item = Attempt;

    fn next(&mut self) -> Option<Self::Item> {
        let prefix = self.pending.pop()?;
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
                let selected = prefix
                    .get(steps.len())
                    .cloned()
                    .unwrap_or_else(|| domain.alternatives[0].clone());
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
        // Ancestors already have queued siblings. Branch only at decisions first
        // encountered by this replay. Reverse insertion gives a depth-first walk;
        // source order has no performance meaning.
        for depth in prefix.len()..steps.len() {
            let step = &steps[depth];
            for alternative in step.domain.alternatives.iter().rev() {
                if *alternative == step.selected {
                    continue;
                }
                let mut sibling: Vec<_> = steps[..depth]
                    .iter()
                    .map(|step| step.selected.clone())
                    .collect();
                sibling.push(alternative.clone());
                self.pending.push(sibling);
            }
        }
        Some(Attempt { steps, result })
    }
}

/// Replay a saved path against freshly derived domains. This validates decision
/// applicability and consumption, not program identity or performance optimality.
pub fn replay(request: Specialization<'_>, expected: &[DecisionRecord]) -> Result<LoweredIr, String> {
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
