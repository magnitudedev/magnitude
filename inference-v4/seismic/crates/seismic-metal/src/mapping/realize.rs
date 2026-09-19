//! `realize`: the instantiated execution IR plus one deterministic rule per former
//! backend decision (spec 8.5, 14.2). Nothing here is searched or ranked.
//!
//!   dispatch       one subgroup per piece; pieces per threadgroup = ceil(max pieces / 65535);
//!                  tiling is already in the instantiated IR
//!   load           Borrow when the borrow proof holds, else Materialize. The mode written by
//!                  instantiation is a placeholder of the IR node, not a selection, and the
//!                  proof reads it: re-executing a load in a loop ends its own loan only when
//!                  that load borrows. The rule is therefore the greatest fixpoint of the
//!                  proof: every load starts as Borrow, a load whose proof fails becomes
//!                  Materialize, until stable (failures only remove aliases, so it is monotone)
//!   tile storage   threadgroup-shared for matrix-intrinsic operands, for a tile of at least one
//!                  subgroup of elements that other lanes read (written cooperatively, one
//!                  share per lane; replicated, every lane would compute every element), or
//!                  when cooperation leaves no private placement; else private: distributed
//!                  from one element per lane (a subgroup of elements) upward when admitted,
//!                  else replicated
//!   reduction      lane-local when admitted and the output exceeds one subgroup; else
//!                  collective (per-lane partial, then one subgroup reduction) when admitted,
//!                  which requires a contract that permits reassociation (`unordered=true` in
//!                  a body with `unordered=true`) over a lane-distributed input; else ordered (authored
//!                  ascending order, bit-exact); else the first admitted algorithm
//!   allocation     always a new slot: no reuse, hence no optional barrier
//!   transfer       the widest exact packed vector the transfer admits
//!   traversal      unroll width 1
use super::{Limits, MAX_GROUPS};
use crate::execution::{prepare_with_transfers, Config, Execution, SUBGROUP};
use crate::reduction::Algorithm;
use seismic_compiler::selection::SelectionError;
use seismic_lang::exec::ir::{LoadMode, StmtKind};
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_realization::dispatch::TilePlacement;

fn reduction_algorithm(
    domain: &crate::reduction::ReductionDomain,
    allow_numerical_effects: bool,
) -> Result<Algorithm, String> {
    let admitted = domain.algorithms();
    if domain.output_capacity() > SUBGROUP as u64 && admitted.contains(&Algorithm::LaneLocal) {
        Ok(Algorithm::LaneLocal)
    } else if allow_numerical_effects && admitted.contains(&Algorithm::Collective) {
        Ok(Algorithm::Collective)
    } else if admitted.contains(&Algorithm::Ordered) {
        Ok(Algorithm::Ordered)
    } else {
        admitted.first().copied().ok_or_else(|| "reduction admits no algorithm".to_string())
    }
}

/// Largest piece count of any root `parallel` statement.
fn max_pieces(lowered: &LoweredIr) -> Result<u64, SelectionError> {
    let mut most = 1u64;
    for statement in &lowered.body {
        let StmtKind::Parallel { extents, .. } = &statement.kind else { continue };
        let pieces = extents.iter().try_fold(1u64, |acc, e| acc.checked_mul(u64::try_from(e.as_constant()?).ok()?));
        most = most.max(pieces.ok_or_else(|| SelectionError::Reconstruction(format!("`{}`: a root parallel extent is not a concrete piece count", lowered.name)))?);
    }
    Ok(most)
}

pub(super) fn realize(
    limits: &Limits,
    lowered: &LoweredIr,
    allow_numerical_effects: bool,
) -> Result<Execution, SelectionError> {
    let mut selected = lowered.clone();
    seismic_compiler::load_rule(&mut selected).map_err(|reason| SelectionError::UnsupportedMapping(format!("`{}`: {reason}", lowered.name)))?;
    let lowered = &selected;
    let pieces = max_pieces(lowered)?;
    let items_per_group = limits.items_per_group(pieces);
    if items_per_group > limits.max_items_per_group() || pieces > limits.max_pieces() {
        return Err(SelectionError::IncompatibleComposition(format!(
            "`{}`: {pieces} pieces need {items_per_group} pieces per threadgroup across {MAX_GROUPS} threadgroups; {} threads per threadgroup admit {}",
            lowered.name,
            limits.max_threads_per_threadgroup,
            limits.max_items_per_group()
        )));
    }
    let integer = |value: u64, what: &str| i64::try_from(value).map_err(|_| SelectionError::IncompatibleComposition(format!("`{}`: {what} {value} exceeds the signed range", lowered.name)));
    let config = Config {
        sg_per_tg: integer(items_per_group, "pieces per threadgroup")?,
        max_threads_per_threadgroup: integer(limits.max_threads_per_threadgroup, "thread capacity")?,
        max_threadgroup_bytes: integer(limits.max_threadgroup_bytes, "threadgroup memory")?,
    };
    let execution = prepare_with_transfers(
        lowered,
        config,
        &mut |_, site| Ok(if site.can_borrow { LoadMode::Borrow } else { LoadMode::Materialize }),
        &mut |decision| {
            let admitted = |placement: &TilePlacement| decision.alternatives.contains(placement);
            // A tile of the outer owner of a launch with inner owner regions is one array per
            // threadgroup, staged cooperatively by all its threads and shared by its SIMD groups.
            if decision.outer_owner {
                return if admitted(&TilePlacement::GroupWide) { Ok(TilePlacement::GroupWide) } else { Err(format!("outer-owner tile `{}` admits no threadgroup-wide placement", decision.name)) };
            }
            let private =if decision.capacity >= SUBGROUP && admitted(&TilePlacement::Distributed) {
                Some(TilePlacement::Distributed)
            } else {
                [TilePlacement::Replicated, TilePlacement::Distributed].into_iter().find(|p| admitted(p))
            };
            // A tile of at least one subgroup of elements that other lanes read is written
            // cooperatively into threadgroup memory: replicating it would make every lane
            // compute every element.
            let cooperative_read = decision.cross_lane_read && decision.capacity >= SUBGROUP && admitted(&TilePlacement::GroupShared);
            match private.filter(|_| !decision.intrinsic_operand && !cooperative_read) {
                Some(placement) => Ok(placement),
                None if admitted(&TilePlacement::GroupShared) => Ok(TilePlacement::GroupShared),
                None => Err(format!("tile `{}` admits no storage placement", decision.name)),
            }
        },
        &mut |decision| reduction_algorithm(&decision.domain, allow_numerical_effects),
        &mut |choice| Ok(choice.new_slot),
        &mut |choice| Ok(choice.maximum.max(1)),
        &mut |_| Ok(1),
    )
    .map_err(|reason| {
        let capacity = ["capacity", "exceed", "limit", "threadgroup"].iter().any(|word| reason.contains(word));
        if reason.contains("no static") || reason.contains("not static") {
            SelectionError::AnalysisUnavailable(format!("`{}`: {reason}", lowered.name))
        } else if capacity {
            SelectionError::IncompatibleComposition(format!(
                "`{}`: {reason} ({pieces} pieces, {items_per_group} per threadgroup, {} threads and {} threadgroup bytes available)",
                lowered.name, limits.max_threads_per_threadgroup, limits.max_threadgroup_bytes
            ))
        } else {
            SelectionError::UnsupportedMapping(format!("`{}`: {reason}", lowered.name))
        }
    })?;
    // The thread stack holds every private array a kernel declares.
    for (launch, memory) in execution.memory().launches().iter().enumerate() {
        if memory.declared_private_bytes_per_lane > limits.max_private_bytes {
            return Err(SelectionError::IncompatibleComposition(format!(
                "`{}` launch {launch}: {} bytes of private arrays per thread exceed the {} byte thread stack limit",
                lowered.name, memory.declared_private_bytes_per_lane, limits.max_private_bytes
            )));
        }
    }
    Ok(execution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::types::DType;

    #[test]
    fn strict_precision_keeps_unordered_sum_on_the_reference_order() {
        let domain = crate::reduction::ReductionDomain::new(
            &[32],
            0,
            DType::F32,
            false,
            TilePlacement::Distributed,
            32,
        )
        .unwrap();
        assert!(domain.algorithms().contains(&Algorithm::Collective));
        assert_eq!(reduction_algorithm(&domain, false).unwrap(), Algorithm::Ordered);
        assert_eq!(reduction_algorithm(&domain, true).unwrap(), Algorithm::Collective);
    }
}
