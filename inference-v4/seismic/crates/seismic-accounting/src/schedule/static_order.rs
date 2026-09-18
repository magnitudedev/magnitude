//! Joint timing and loop-preserving instruction-order feasibility.
//!
//! A compiler controls one order per basic block, not a separate hardware issue
//! permutation on each loop visit. This constraint existentially represents every
//! legal block order without enumerating their factorial product. Timings remain
//! conditional model events; extracting an order does not command native cycles.
use super::{Model, Schedule};
use cranelift_codegen::ir::{Block, Inst};
use seismic_realization::scheduling::BlockOrder;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Visit {
    pub invocation: u64,
    pub occurrence: u64,
    /// Primitive issue roots, aligned with `Constraint::instructions`.
    pub roots: Vec<Vec<usize>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub block: Block,
    pub instructions: Vec<Inst>,
    /// Legal static precedence, indexed in `instructions`.
    pub predecessors: Vec<(usize, usize)>,
    pub visits: Vec<Visit>,
}

pub fn validate(model: &Model) -> Result<(), String> {
    let mut blocks = BTreeSet::new();
    let mut instructions = BTreeSet::new();
    let mut roots = BTreeSet::new();
    for constraint in &model.static_orders {
        if !blocks.insert(constraint.block) || constraint.instructions.is_empty() {
            return Err("static-order blocks must be unique and nonempty".into());
        }
        for &instruction in &constraint.instructions {
            if !instructions.insert(instruction) {
                return Err("static-order instruction occurs in multiple positions".into());
            }
        }
        let count = constraint.instructions.len();
        let mut edges = BTreeSet::new();
        for &(before, after) in &constraint.predecessors {
            if before >= count
                || after >= count
                || before == after
                || !edges.insert((before, after))
            {
                return Err("invalid static-order predecessor".into());
            }
        }
        if topological(count, &edges).is_none() {
            return Err("cyclic static-order predecessors".into());
        }
        let mut visits = BTreeSet::new();
        for visit in &constraint.visits {
            if !visits.insert((visit.invocation, visit.occurrence)) || visit.roots.len() != count {
                return Err("invalid static-order visit identity or arity".into());
            }
            for group in &visit.roots {
                if group.is_empty() {
                    return Err("static instruction has no issue roots".into());
                }
                for &root in group {
                    if root >= model.operations.len() || !roots.insert(root) {
                        return Err("invalid or repeated static instruction issue root".into());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Unknown start times relax constraints. A direction is excluded only by an
/// assigned pair that contradicts it; no guessed timing participates in pruning.
pub fn fits(model: &Model, starts: &[Option<u64>]) -> Result<bool, String> {
    if starts.len() != model.operations.len() {
        return Err("static-order schedule arity differs from model".into());
    }
    for constraint in &model.static_orders {
        if resolve(constraint, starts)?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Independently checks the complete timing witness before selecting its common
/// static orders. Equal-time ties use source index solely for determinism.
pub fn orders(model: &Model, schedule: &Schedule) -> Result<Vec<BlockOrder>, String> {
    model.check_schedule(schedule)?;
    let starts: Vec<_> = schedule.starts.iter().copied().map(Some).collect();
    model
        .static_orders
        .iter()
        .map(|constraint| {
            let order =
                resolve(constraint, &starts)?.ok_or("schedule has no common static order")?;
            Ok(BlockOrder {
                block: constraint.block,
                instructions: order
                    .into_iter()
                    .map(|index| constraint.instructions[index])
                    .collect(),
            })
        })
        .collect()
}

fn resolve(constraint: &Constraint, starts: &[Option<u64>]) -> Result<Option<Vec<usize>>, String> {
    let mut edges: BTreeSet<_> = constraint.predecessors.iter().copied().collect();
    let count = constraint.instructions.len();
    for left in 0..count {
        for right in left + 1..count {
            let mut left_first = true;
            let mut right_first = true;
            for visit in &constraint.visits {
                let left_roots = visit
                    .roots
                    .get(left)
                    .ok_or("invalid static-order visit arity")?;
                let right_roots = visit
                    .roots
                    .get(right)
                    .ok_or("invalid static-order visit arity")?;
                for &a in left_roots {
                    for &b in right_roots {
                        let a = starts.get(a).ok_or("invalid static-order issue root")?;
                        let b = starts.get(b).ok_or("invalid static-order issue root")?;
                        if let (Some(a), Some(b)) = (a, b) {
                            left_first &= a <= b;
                            right_first &= b <= a;
                        }
                    }
                }
            }
            match (left_first, right_first) {
                (false, false) => return Ok(None),
                (true, false) => {
                    edges.insert((left, right));
                }
                (false, true) => {
                    edges.insert((right, left));
                }
                (true, true) => {}
            }
        }
    }
    Ok(topological(count, &edges))
}
fn topological(count: usize, edges: &BTreeSet<(usize, usize)>) -> Option<Vec<usize>> {
    let mut incoming = vec![0usize; count];
    let mut followers = vec![Vec::new(); count];
    for &(before, after) in edges {
        if before >= count || after >= count {
            return None;
        }
        incoming[after] += 1;
        followers[before].push(after);
    }
    let mut ready: BTreeSet<_> = (0..count).filter(|&i| incoming[i] == 0).collect();
    let mut order = Vec::with_capacity(count);
    while let Some(next) = ready.pop_first() {
        order.push(next);
        for &after in &followers[next] {
            incoming[after] -= 1;
            if incoming[after] == 0 {
                ready.insert(after);
            }
        }
    }
    (order.len() == count).then_some(order)
}
