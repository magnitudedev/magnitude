//! `realize`: the instantiated execution IR plus one deterministic rule per former backend
//! decision (spec 8.5, 14.2). Nothing here is searched or ranked.
//!
//!   participation  one thread per piece; one warp of 32 lanes per piece exactly when the
//!                  instantiated body names a participant intrinsic (lane index, shuffle, sum)
//!   load           Borrow when the borrow proof holds, else Materialize: the greatest
//!                  fixpoint of the proof starting from all-borrow (a load's own mode is an
//!                  input of the proof; failures only remove aliases, so it is monotone)
//!   phases         root `parallel` statements are launches over their pieces; every run of
//!                  other root statements is one single-thread launch; values crossing a
//!                  launch are published to invocation-owned buffers (shared phase plan)
//!   tile storage   every tile and materialized snapshot is a bump allocation of the owner
//!                  thread's scratch in device global memory, never reused
//!   reduction      the owner thread folds in authored ascending order (bit-exact against
//!                  the interpreter); a reassociation permission is not used
//!   block size     items per block = min(work items, 256 threads / lanes per item), raised
//!                  to ceil(work items / grid blocks) when one grid row cannot hold the launch;
//!                  the grid is one-dimensional
//!   math           exp/log/sin/cos link the bundled PTX libm port (THIRD_PARTY.md)
//!   floating point FMA appears only where authored; PTX has no implicit contraction
use super::Limits;
use crate::execution::{Execution, Launches};
use seismic_compiler::selection::SelectionError;
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_realization::dispatch::Participation;
use seismic_realization::{CallConv, Dispatch};

/// Threads of one block when neither the work nor the grid says otherwise. A multiple of
/// the warp width. Conventional; its effect on GB10 is unmeasured.
pub const BLOCK_THREADS: u32 = 256;

/// Threads of one block for a launch of `work_items` items of `lanes` lanes (block-size rule).
pub(super) fn threads_per_block(
    limits: &Limits,
    work_items: u64,
    lanes: u64,
) -> Result<u32, String> {
    if lanes == 0 || u64::from(limits.max_threads_per_block) < lanes {
        return Err(format!(
            "a block of {} threads cannot hold one work item of {lanes} lanes",
            limits.max_threads_per_block
        ));
    }
    let preferred = u64::from(BLOCK_THREADS.min(limits.max_threads_per_block)) / lanes;
    let items = work_items
        .clamp(1, preferred.max(1))
        .max(work_items.div_ceil(u64::from(limits.max_grid_x.max(1))));
    let threads = items
        .checked_mul(lanes)
        .filter(|t| *t <= u64::from(limits.max_threads_per_block));
    threads.and_then(|t| u32::try_from(t).ok()).ok_or_else(|| {
        format!(
            "{work_items} work items of {lanes} lanes exceed a grid of {} blocks of {} threads",
            limits.max_grid_x, limits.max_threads_per_block
        )
    })
}

pub(super) fn realize(
    limits: &Limits,
    target_profile: &crate::target::TargetProfile,
    lowered: &LoweredIr,
) -> Result<Launches, SelectionError> {
    let mut selected = lowered.clone();
    seismic_compiler::scalar_load_rule(&mut selected).map_err(|reason| {
        SelectionError::UnsupportedMapping(format!("`{}`: {reason}", selected.name))
    })?;
    let participation = if seismic_compiler::subgroup_required(&selected) {
        if limits.warp_size != 32 {
            return Err(SelectionError::UnsupportedMapping(format!(
                "`{}`: participant intrinsics need a 32-lane warp; the device has {}",
                selected.name, limits.warp_size
            )));
        }
        Participation::Subgroup { lanes: 32 }
    } else {
        Participation::Thread
    };
    let sequence = seismic_compiler::scalar_sequence_participants_resolved(
        &selected,
        CallConv::SystemV,
        Dispatch::ParallelRoot,
        participation,
    )
    .map_err(|reason| {
        if reason.contains("static") {
            SelectionError::AnalysisUnavailable(format!("`{}`: {reason}", selected.name))
        } else {
            SelectionError::UnsupportedMapping(format!(
                "`{}`: scalar realization: {reason}",
                selected.name
            ))
        }
    })?;
    if sequence.phases.is_empty() {
        return Err(SelectionError::Reconstruction(format!(
            "`{}`: the phase plan has no launch",
            selected.name
        )));
    }
    let mut phases = Vec::with_capacity(sequence.phases.len());
    for (ordinal, phase) in sequence.phases.into_iter().enumerate() {
        let program = phase.program;
        let lanes = u64::from(program.participation.lanes());
        let threads = threads_per_block(limits, program.work_items, lanes).map_err(|reason| {
            SelectionError::IncompatibleComposition(format!(
                "`{}` launch {ordinal}: {reason}",
                selected.name
            ))
        })?;
        let target = target_profile
            .plan(crate::target::TargetRequirement::ScalarBaseline)
            .map_err(|reason| {
                SelectionError::UnsupportedMapping(format!(
                    "`{}` launch {ordinal}: CUDA target: {reason}",
                    selected.name
                ))
            })?;
        let execution = Execution::new_for_target(program, threads, limits.launch(), target)
            .map_err(|reason| {
                SelectionError::UnsupportedMapping(format!(
                    "`{}` launch {ordinal}: PTX: {reason}",
                    selected.name
                ))
            })?;
        // Hard limits are rechecked on the realized launch.
        let storage = execution.storage();
        let invocation_bytes =
            (storage.scratch_bytes as u64).checked_add(storage.status_bytes as u64);
        if invocation_bytes.is_none_or(|bytes| bytes > limits.max_scratch_bytes) {
            return Err(SelectionError::IncompatibleComposition(format!(
                "`{}` launch {ordinal}: {} scratch bytes of {} threads exceed the {} byte invocation scratch limit",
                selected.name,
                storage.scratch_bytes,
                execution.dispatch().participating_lanes(),
                limits.max_scratch_bytes
            )));
        }
        phases.push(execution);
    }
    Ok(Launches {
        name: selected.name,
        phases,
    })
}
