//! Unified Path 0 (`RAM -> VRAM`) link controller ([`LinkMover`]):
//! owns every asynchronous copy over the discrete GPU transfer queue
//! (`experts::copies`), unifying permanent promotions, write-backs, and
//! zero-spin opportunistic DMA staging under one priority-ordered, cancellable
//! controller:
//!
//! 1. **Write-backs (`TransferPriority::WriteBack`)**: release bank records
//!    that promotions and prompt leases wait for.
//! 2. **Opportunistic DMA staging (`TransferPriority::Stage { step, layer, p }`)**:
//!    reuses the lender group's idle `0..lent` bank records as an even/odd
//!    double-buffered staging ring owned directly by `BankRecords`
//!    (`Record::Staging`), fed by the shared `NameLedger`
//!    (`Ledger::schedule_for_path(PathKind::Link, ...)`) and `DraftWindow`
//!    carryover on BOTH 100%-RAM and unsettled 3-tier workloads. Each expert's
//!    copy batch ends with `TransferItem::Signal` writing `MAP_STAGE` on the
//!    copy engine (zero GPU spin / zero driver-580 deadlock by construction).
//!    Telemetry is tracked separately (`Telemetry::staged_link`) so
//!    `POLICY_LINK_BUSY_NS` and promotion earning are never polluted.
//! 3. **Permanent promotions (`TransferPriority::Promote`)**: fills remaining
//!    copy-engine depth with `Reuse`-ranked promotions into the permanent bank,
//!    with reserve sizing from the decode miss curve.

use super::Shared;
use super::core::Core;
use crate::assessment::TierCopyCosts;
use crate::experts::copies::{BatchItem, Run, Signal, TransferPriority};
use crate::experts::directory::{BankRecord, HostRecord, MissCurve, Point};
use crate::experts::interest::{Act, Admit, Class, Ledger, Read as ReadState, Scheduled};
use crate::experts::map::stage_word;
use crate::experts::staging::Staged;
use crate::experts::watcher::Step;
use magnitude_cost_model::{LinkIdleBudget, PathAdmission, PathKind};
use magnitude_kernels::constants::{MAP_STAGE, MAP_WORDS};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// What Path 0 copies with: the device's copy engine. Destination bank record
/// views are resolved on demand from `Core::groups[group].bank`
/// (`Records::pooled`), so idle bank slabs hold no view and the pool can
/// vacate their blocks on shrink or reclaim.
#[derive(Clone)]
pub(in crate::experts) struct Copier {
    pub copies: crate::experts::copies::Copies,
}

/// Admission wrapper for Link-path staging moves inside
/// `Ledger::schedule_for_path(PathKind::Link, ...)`: consumes from the shared
/// `LinkIdleBudget` as moves are admitted in priority order.
struct LinkAdmitting<'a> {
    admission: PathAdmission,
    bytes: u64,
    idle: &'a Cell<LinkIdleBudget>,
}

impl Admit for LinkAdmitting<'_> {
    fn admits(&self, p: f64) -> bool {
        let mut budget = self.idle.get();
        let admitted = self.admission.admits(p, self.bytes, budget.remaining());
        if admitted {
            budget.consume(self.admission.transfer_seconds(self.bytes));
            self.idle.set(budget);
        }
        admitted
    }
}

/// Unified Path 0 (`RAM -> VRAM`) link mover: manages write-backs,
/// `Reuse`-ranked permanent promotions, and `NameLedger`-driven opportunistic
/// DMA staging on `Copies`.
///
/// Permanent VRAM residency stays at full `Directory::capacity` (`reserve = 0`):
/// `Directory::promotions()` releases victims on demand (`want_releases =
/// short.min(budget_left)`) whenever `free == 0` without ever pre-emptively
/// emptying held VRAM slots into an idle reserve pool.
pub(super) struct LinkMover {
    /// Issue timestamps of in-flight opportunistic staging copies `(group, slot) -> (target_step, issued_at)`
    /// so `settle` can report landed link-staged hints to `NameLedger::link_landed`.
    stage_issued: HashMap<(usize, usize), (Step, Instant)>,
    /// Total opportunistic staging transfers landed (for one-time activation telemetry log).
    staged_total: u64,
}

impl LinkMover {
    pub(super) fn new() -> Self {
        Self { stage_issued: HashMap::new(), staged_total: 0 }
    }

    /// Settle every completed write-back, permanent promotion, and opportunistic
    /// staging copy, and drain releases and landed staging slots past the
    /// frontier. Permanent moves report to `Telemetry::moved`; opportunistic
    /// staging reports separately to `Telemetry::staged_link`.
    pub(super) fn settle(&mut self, core: &mut Core, ledger: &mut Ledger, shared: &Shared, copier: &Copier, progress: Point) -> bool {
        let mut moved = false;
        if let Some(copy) = line(core, shared) {
            let bytes = record_bytes(core);

            // 1. Write-backs completed.
            for (group, record, batch) in core.directory.backing() {
                match copier.copies.completed(batch) {
                    Ok(false) => {}
                    Ok(true) => {
                        core.directory.written_back(group, record, true);
                        shared.telemetry.moved(copy.back.transfer_seconds(bytes[group] as f64), 1);
                        moved = true;
                    }
                    Err(error) => {
                        core.directory.written_back(group, record, false);
                        shared.end_serving(format!("a release's write-back into the host tier: {error}"));
                        moved = true;
                    }
                }
            }

            // 2. Permanent promotions completed.
            let mut completed = Vec::new();
            for (group, record, batch) in core.directory.copying() {
                match copier.copies.completed(batch) {
                    Ok(false) => {}
                    Ok(true) => completed.push((group, record)),
                    Err(error) => {
                        core.directory.unpromoted(group, record);
                        shared.end_serving(format!("a promotion's copy into the bank: {error}"));
                        moved = true;
                    }
                }
            }
            if !completed.is_empty() {
                let link = completed.iter().map(|&(group, _)| copy.transfer.transfer_seconds(bytes[group] as f64)).sum::<f64>();
                shared.telemetry.moved(link, completed.len());
                for &(group, record) in &completed {
                    shared
                        .transport
                        .try_release_hint_slot(crate::experts::transport::DestSlot { group, slot: record.index() });
                }
                core.directory.promoted(completed, progress, &core.steps);
                moved = true;
            }

            // 3. Opportunistic staging copies completed (separated telemetry).
            let mut staged_bytes = 0u64;
            let mut staged_count = 0usize;
            let now = Instant::now();
            for (group, slot, batch) in core.directory.staging_copying() {
                match copier.copies.completed(batch) {
                    Ok(false) => {}
                    Ok(true) => {
                        let (layer, expert) = core.directory.staged_landed(group, slot);
                        if let Some((target_step, issued_at)) = self.stage_issued.remove(&(group, slot)) {
                            ledger.link_landed(target_step, layer, expert, issued_at, now);
                        }
                        staged_bytes += bytes[group];
                        staged_count += 1;
                        moved = true;
                    }
                    Err(error) => {
                        self.stage_issued.remove(&(group, slot));
                        core.directory.unstage(group, slot);
                        shared.end_serving(format!("an opportunistic staging copy into the bank: {error}"));
                        moved = true;
                    }
                }
            }
            if staged_count > 0 {
                let link = copy.transfer.transfer_seconds(staged_bytes as f64);
                shared.telemetry.staged_link(link, staged_count);
                let was = self.staged_total;
                self.staged_total = self.staged_total.saturating_add(staged_count as u64);
                if was == 0 {
                    shared.time.lock().update_residency(|state| {
                        state.link_staging = true;
                    });
                    tracing::info!(
                        staged_count,
                        staged_bytes,
                        "LinkMover async DMA staging ring active (MAP_STAGE)"
                    );
                }
            }
        }

        // 4. Drain releases (including stamped-unmap releases without write-back),
        //    deferred demand landings, and landed staging slots no longer pinned
        //    by any joined demand consumer in `ExpertTransport`.
        moved |= core.directory.drain(progress, &core.steps);
        moved |= core.directory.drain_demand_landings(|group, slot| {
            shared
                .transport
                .can_reuse_slot(crate::experts::transport::DestSlot { group, slot })
        });
        moved |= core.directory.drain_staging_with(progress, &core.steps, |group, slot| {
            shared
                .transport
                .try_release_hint_slot(crate::experts::transport::DestSlot { group, slot })
        });
        moved
    }

    /// Cancel every unbegun opportunistic staging batch on `copier`, freeing
    /// its staging slot and host-tier hold immediately (used when a resize
    /// waits for in-flight copies to drain, matching `Prefetch::cancel_cover`).
    pub(super) fn cancel_unbegun(&mut self, core: &mut Core, copier: &Copier) -> bool {
        let mut cancelled = false;
        for (group, slot, batch) in core.directory.staging_copying() {
            if copier.copies.cancel(batch) {
                self.stage_issued.remove(&(group, slot));
                core.directory.unstage(group, slot);
                cancelled = true;
            }
        }
        cancelled
    }

    /// One turn of Path 0: size promotion reserves, settle in-flight copies,
    /// then dispatch in strict priority order:
    /// 1. Write-backs (`TransferPriority::WriteBack`)
    /// 2. Predicted & `DraftWindow` opportunistic staging (`TransferPriority::Stage`)
    /// 3. Permanent promotions (`TransferPriority::Promote`)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn advance(
        &mut self,
        core: &mut Core,
        ledger: &mut Ledger,
        shared: &Shared,
        copier: &Copier,
        _decoded: Option<&MissCurve>,
        held: bool,
        progress: Point,
    ) -> bool {
        let mut moved = self.settle(core, ledger, shared, copier, progress);
        let Some(copy) = line(core, shared) else {
            return moved;
        };
        if held || shared.life.ended().is_some() {
            return moved;
        }
        let bytes = record_bytes(core);

        // Priority 1: Write-backs (free bank records promotions/leases wait for).
        // Depth excludes in-flight opportunistic `Stage` batches so already-
        // queued staging copies never lock out write-backs.
        for back @ (group, record, ..) in core.directory.backs_queued() {
            let depth = permanent_post_depth(copy.depth(bytes[group]), copier.copies.flight(), core.directory.staging_copying().len());
            match post_back(core, copier, depth, back) {
                Ok(true) => moved = true,
                Ok(false) => break,
                Err(error) => {
                    core.directory.written_back(group, record, false);
                    shared.end_serving(format!("a release's write-back into the host tier: {error}"));
                    return true;
                }
            }
        }

        // Priority 2: Permanent VRAM promotions (save every future token,
        // strictly higher value than 1-step opportunistic `MAP_STAGE`). Run
        // BEFORE `stage_predicted` and exclude in-flight `Stage` batches from
        // `ask()`'s budget and `post_promote()`'s depth cap so opportunistic
        // staging can never starve permanent VRAM residency growth.
        // Pause all promotions while a prompt group is running on
        // `tier_copy_dma` so prefill async DMA (`TransferPriority::PrefillFill`)
        // never contends with or deadlocks behind `Promoting` batches.
        let pause_promote = core.extents.tier_copy_dma && core.steps.prompt_running();
        let lending = core.steps.lending();
        if lending || pause_promote {
            // Cancel any unbegun promotion batches targeting `0..lent` (or any
            // unbegun promotion while an async-DMA prompt runs) so a prompt
            // group never waits on queued background promotions.
            for (group, record, batch) in core.directory.copying() {
                if (pause_promote || core.directory.bank_lends(group, record.index())) && copier.copies.cancel(batch) {
                    core.directory.unpromoted(group, record);
                    moved = true;
                }
            }
        }
        if !pause_promote {
            moved |= self.ask(core, shared, copier, &copy);
            let mut queued = core.directory.queued();
            queued.retain(|&(group, record, (layer, expert))| {
                let kept = !(lending && core.directory.bank_lends(group, record.index()))
                    && core.directory.tier_record(layer, expert).is_some()
                    && core.directory.reuse_now().in_top(layer, expert);
                if !kept {
                    core.directory.unpromoted(group, record);
                    moved = true;
                }
                kept
            });
            for promotion @ (group, record, _) in queued {
                let depth = permanent_post_depth(copy.depth(bytes[group]), copier.copies.flight(), core.directory.staging_copying().len());
                match post_promote(core, shared, copier, depth, promotion) {
                    Ok(true) => moved = true,
                    Ok(false) => break,
                    Err(error) => {
                        core.directory.unpromoted(group, record);
                        shared.end_serving(format!("a promotion's copy into the bank: {error}"));
                        return true;
                    }
                }
            }
        }

        // Priority 3: Opportunistic staging from NameLedger + DraftWindow
        // carryover (works on BOTH 100%-RAM and unsettled 3-tier workloads!).
        moved |= self.stage_predicted(core, ledger, shared, copier, &copy, &bytes, progress);
        moved
    }

    /// Schedule opportunistic DMA staging over every partial group's dedicated
    /// even/odd double-buffered staging ring (`BankRecords::stage_rings`,
    /// completely independent of `lent == 0` and prompt scratch leases) from
    /// `NameLedger::schedule_for_path(PathKind::Link, ...)` and `DraftWindow`
    /// carryover. No `group_tier_settled` blackout: any `Located::Kept` Tier-1
    /// expert absent from Tier 0 is eligible on both 100%-RAM and unsettled
    /// 3-tier workloads.
    #[allow(clippy::too_many_arguments)]
    fn stage_predicted(
        &mut self,
        core: &mut Core,
        ledger: &mut Ledger,
        shared: &Shared,
        copier: &Copier,
        copy: &TierCopyCosts,
        bytes: &[u64],
        progress: Point,
    ) -> bool {
        if core.steps.prompt_running() {
            return false;
        }
        let (frontier, next_step) = core.steps.unbegun_beside_frontier(|frontier, at| (frontier, at.next()));
        let mut moved = core.directory.drain_staging_with(progress, &core.steps, |group, slot| {
            shared
                .transport
                .try_release_hint_slot(crate::experts::transport::DestSlot { group, slot })
        });

        let admission = shared.telemetry.link_admission(Some(copy.transfer));
        let layers = core.layers.len();
        let c_layer = if layers == 0 { 0.0 } else { admission.terms.compute / layers as f64 };
        // Span the remaining compute window of the active verify/decode burst
        // (`remaining_layers * c_layer`) so Path 0 keeps the PCIe copy engine
        // saturated across the full burst rather than only 2 layers.
        let remaining_layers = layers.saturating_sub(progress.layer.saturating_add(1)).max(1);
        let c_window = (remaining_layers as f64) * c_layer;
        let draft_idle = ledger.draft_window_remaining(Instant::now());
        let wanted_all = ledger.wanted();

        for group in 0..core.groups.len() {
            let ring_half = core.directory.stage_ring_half(group);
            if ring_half == 0 || core.directory.group_whole(group) {
                continue;
            }
            let record_bytes = bytes[group];
            if record_bytes == 0 {
                continue;
            }
            // Opportunistic `Stage` batches are priority-ordered by
            // `(step, layer, -p)` and cancellable while waiting in `Copies`,
            // so allow queuing up to the parity-half capacity (at least
            // `copy.depth`) so all hints for `l+1..=l+HINT_LEADS` stream back-
            // to-back at PCIe line rate without per-pair host wakeups.
            let stage_depth = copy.depth(record_bytes).max(ring_half);
            let queued_batches = core.directory.copying().len() + core.directory.backing().len() + core.directory.staging_copying().len();
            let queued_on_link = queued_batches as f64 * copy.transfer.transfer_seconds(record_bytes as f64);
            let idle = Cell::new(LinkIdleBudget::new(c_window, draft_idle, queued_on_link));

            // Gather wanted experts in `group` plus any currently-copying staging
            // experts (so fallen/passed unbegun staging batches can be cancelled).
            let mut candidates: Vec<Staged> = wanted_all
                .iter()
                .copied()
                .filter(|&(layer, _)| core.layers[layer].group == group)
                .collect();
            let mut seen: HashSet<Staged> = candidates.iter().copied().collect();
            for (g, slot, _) in core.directory.staging_copying() {
                if g == group {
                    if let Some(crate::experts::records::Record::Staging { staged, .. }) = core.directory.bank_record(g, slot) {
                        if seen.insert(staged) {
                            candidates.push(staged);
                        }
                    }
                }
            }

            if !candidates.is_empty() {
                let states: HashMap<Staged, ReadState> = candidates
                    .iter()
                    .map(|&staged @ (layer, expert)| {
                        let state = if core.directory.held(layer, expert).is_some()
                            || core.directory.placing(staged)
                            || core.directory.tier_record(layer, expert).is_none()
                        {
                            ReadState::Held
                        } else if let Some((_, _, batch)) = core.directory.staged_entry(group, staged) {
                            match batch {
                                Some(ticket) => ReadState::Reading { begun: copier.copies.is_begun(ticket) },
                                None => ReadState::Held,
                            }
                        } else {
                            ReadState::Absent
                        };
                        (staged, state)
                    })
                    .collect();

                let scheduled = ledger.schedule_for_path(
                    PathKind::Link,
                    candidates,
                    |layer, expert| states[&(layer, expert)],
                    |_| LinkAdmitting { admission, bytes: record_bytes, idle: &idle },
                );

                for Scheduled { layer, expert, act, .. } in scheduled {
                    let staged = (layer, expert);
                    match act {
                        Act::Keep | Act::Issue(Class::Due) | Act::Reprice(Class::Due) => {}
                        Act::Cancel => {
                            if let Some((slot, _, Some(ticket))) = core.directory.staged_entry(group, staged) {
                                if copier.copies.cancel(ticket) {
                                    self.stage_issued.remove(&(group, slot));
                                    core.directory.unstage(group, slot);
                                    moved = true;
                                }
                            }
                        }
                        Act::Reprice(Class::Expected { step, p }) => {
                            if let Some((_, _, Some(ticket))) = core.directory.staged_entry(group, staged) {
                                copier.copies.reprice(ticket, TransferPriority::Stage { step, layer, p });
                            }
                        }
                        Act::Issue(Class::Expected { step, p }) => {
                            if copier.copies.flight() >= stage_depth {
                                continue;
                            }
                            let target_step = if step < frontier || core.steps.ended(step) { next_step } else { step };
                            let Some(slot) = core.directory.free_stage_slot(group, target_step) else {
                                continue;
                            };
                            let priority = TransferPriority::Stage { step: target_step, layer, p };
                            match post_stage(core, shared, copier, stage_depth, priority, group, slot, target_step, staged) {
                                Ok(Some((tier, ticket))) => {
                                    self.stage_issued.insert((group, slot), (target_step, Instant::now()));
                                    core.directory.stage(group, slot, staged, target_step, tier, ticket, (target_step & 1) as u8);
                                    moved = true;
                                }
                                Ok(None) => break,
                                Err(error) => {
                                    shared.end_serving(format!("an opportunistic staging copy into the bank: {error}"));
                                    return true;
                                }
                            }
                        }
                    }
                }
            }

            // Cross-burst & in-verify carryover on Link: both during MTP
            // `DraftWindow` (`draft_idle > 0.0`, targeting `next_step`) and
            // during active verify compute (`c_window` slack, targeting
            // remaining layers `l+1..L-1` of the active step first and early
            // layers `0..=l` of `next_step` second), fill up to half of each
            // parity ring with live-calibrated Tier-1-only `Reuse` carryover
            // experts while keeping >= half the ring free for live within-pass
            // `routed_hint_route` / `Source::Token` hints.
            let carryover_cap = (ring_half / 2).max(1);
            let active_step = ledger.latest_active_step().filter(|&s| s >= frontier && !core.steps.ended(s));
            let in_verify = draft_idle == 0.0 && active_step.is_some();
            let step_for_layer = |layer: usize| {
                if in_verify && layer > progress.layer {
                    active_step.unwrap_or(next_step)
                } else {
                    next_step
                }
            };
            let room_for_step = |directory: &crate::experts::directory::Directory, target_step: Step| {
                carryover_cap.saturating_sub(directory.staged_for_step(group, target_step))
            };
            let max_room = if in_verify {
                let s_cur = active_step.unwrap_or(next_step);
                room_for_step(&core.directory, s_cur).max(room_for_step(&core.directory, next_step))
            } else {
                room_for_step(&core.directory, next_step)
            };
            if (draft_idle > 0.0 || in_verify) && max_room > 0 && copier.copies.flight() < stage_depth && idle.get().remaining() > 0.0 {
                let room = stage_depth.saturating_sub(copier.copies.flight()).min(max_room);
                let mut carryover = {
                    let reuse = core.directory.reuse_now();
                    reuse.next_step_candidates(room, false, |layer, expert| {
                        core.layers[layer].group == group
                            && core.directory.held(layer, expert).is_none()
                            && !core.directory.placing((layer, expert))
                            && core.directory.tier_record(layer, expert).is_some()
                            && core.directory.staged_entry(group, (layer, expert)).is_none()
                    })
                };
                carryover.sort_unstable_by(|a, b| {
                    let a_next = !in_verify || a.layer <= progress.layer;
                    let b_next = !in_verify || b.layer <= progress.layer;
                    a_next
                        .cmp(&b_next)
                        .then_with(|| a.layer.cmp(&b.layer))
                        .then_with(|| b.p_next.total_cmp(&a.p_next))
                });
                for cand in carryover {
                    if copier.copies.flight() >= stage_depth {
                        break;
                    }
                    let target_step = step_for_layer(cand.layer);
                    if room_for_step(&core.directory, target_step) == 0 {
                        continue;
                    }
                    let mut budget = idle.get();
                    let t_bw = admission.transfer_seconds(record_bytes);
                    let free_slack = if draft_idle > 0.0 { budget.remaining() } else { 0.0 };
                    if admission.carryover_net_benefit(cand.p_next, record_bytes, free_slack) <= 0.0 {
                        continue;
                    }
                    let Some(slot) = core.directory.free_stage_slot(group, target_step) else {
                        continue;
                    };
                    let priority = TransferPriority::Stage { step: target_step, layer: cand.layer, p: cand.p_next.clamp(0.0, 1.0) };
                    match post_stage(core, shared, copier, stage_depth, priority, group, slot, target_step, (cand.layer, cand.expert)) {
                        Ok(Some((tier, ticket))) => {
                            self.stage_issued.insert((group, slot), (target_step, Instant::now()));
                            core.directory.stage(group, slot, (cand.layer, cand.expert), target_step, tier, ticket, (target_step & 1) as u8);
                            budget.consume(t_bw);
                            idle.set(budget);
                            moved = true;
                            if t_bw > 0.0 && budget.remaining() <= 0.0 {
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(error) => {
                            shared.end_serving(format!("an opportunistic staging copy into the bank: {error}"));
                            return true;
                        }
                    }
                }
            }
        }

        moved
    }

    /// Ask the ranking for permanent promotions up to remaining permanent
    /// copy-engine headroom (excluding in-flight opportunistic `Stage` copies).
    fn ask(&mut self, core: &mut Core, shared: &Shared, copier: &Copier, copy: &TierCopyCosts) -> bool {
        if core.extents.tier_copy_dma && core.steps.prompt_running() {
            return false;
        }
        let max_depth = record_bytes(core)
            .into_iter()
            .filter(|&bytes| bytes > 0)
            .map(|bytes| copy.depth(bytes))
            .max()
            .unwrap_or(0);
        let budget = permanent_budget(
            max_depth,
            copier.copies.flight(),
            core.directory.staging_copying().len(),
            core.directory.queued().len(),
            core.directory.backs_queued().len(),
        );
        if budget == 0 && !core.directory.has_surrenders() {
            return false;
        }
        if !core.directory.settle_changes() {
            return false;
        }
        let directory = &mut core.directory;
        let asked = core.steps.unbegun_beside_frontier(|_, at| directory.promotions(at, budget));
        for &(group, tier_record) in &asked.surrendered {
            shared.transport.bump_source_generation(group, tier_record);
        }
        shared.promoted.store(directory.asked(), std::sync::atomic::Ordering::Relaxed);
        !asked.promotions.is_empty() || !asked.releases.is_empty()
    }
}

/// Post one expert's opportunistic H2D staging copy runs followed by a
/// `TransferItem::Signal` writing `MAP_STAGE` on the copy engine itself,
/// registered with `ExpertTransport` so decode demand can join the in-flight
/// transfer instead of issuing a duplicate PCIe copy.
#[allow(clippy::too_many_arguments)]
fn post_stage(
    core: &mut Core,
    shared: &Shared,
    copier: &Copier,
    depth: usize,
    priority: TransferPriority,
    group: usize,
    slot: usize,
    target_step: Step,
    (layer, expert): Staged,
) -> Result<Option<(HostRecord, u64)>, String> {
    let (directory, groups, layers) = (&mut core.directory, &core.groups, &core.layers);
    let signal_value = stage_word(target_step, slot);
    let signal_index = (expert as u64) * MAP_WORDS + MAP_STAGE;
    let signal_word = layers[layer].map.tensor().clone();
    let posted = copier.copies.post_tracked_within(depth, priority, || {
        let tier = directory.hold_tier(layer, expert).expect("a selected staging candidate's tier record is kept");
        let transfer_id = shared.transport.register_hint(
            crate::experts::transport::ExpertKey { group, layer, expert },
            tier.0 as usize,
            crate::experts::transport::DestSlot { group, slot },
        );
        let tensors = groups[group].tier.as_ref().expect("opportunistic staging copies from its group's host tier");
        let mut items = Vec::with_capacity(tensors.len() + 1);
        for (index, tensor) in tensors.iter().enumerate() {
            items.push(BatchItem::Run(Run {
                source: tensor.records.tensor(u64::from(tier.0)).clone(),
                source_offset: 0,
                destination: groups[group].bank_view(index, slot),
                destination_offset: 0,
                bytes: tensor.records.bytes(),
            }));
        }
        items.push(BatchItem::Signal(Signal {
            word: signal_word,
            index: signal_index,
            value: signal_value,
        }));
        (items, tier, Some(transfer_id))
    })?;
    Ok(posted)
}

/// Post a permanent promotion's copy batch at `TransferPriority::Promote`,
/// registered with `ExpertTransport` so concurrent decode demand can join the
/// in-flight promotion instead of issuing a duplicate PCIe copy.
fn post_promote(
    core: &mut Core,
    shared: &Shared,
    copier: &Copier,
    depth: usize,
    (group, record, (layer, expert)): (usize, BankRecord, Staged),
) -> Result<bool, String> {
    let (directory, groups) = (&mut core.directory, &core.groups);
    let posted = copier.copies.post_tracked_within(depth, TransferPriority::Promote, || {
        let tier = directory.hold_tier(layer, expert).expect("a queued promotion's tier record is kept");
        let transfer_id = shared.transport.register_hint(
            crate::experts::transport::ExpertKey { group, layer, expert },
            tier.0 as usize,
            crate::experts::transport::DestSlot { group, slot: record.index() },
        );
        let tensors = groups[group].tier.as_ref().expect("a promotion copies from its group's host tier");
        let runs = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                BatchItem::Run(Run {
                    source: tensor.records.tensor(u64::from(tier.0)).clone(),
                    source_offset: 0,
                    destination: groups[group].bank_view(index, record.index()),
                    destination_offset: 0,
                    bytes: tensor.records.bytes(),
                })
            })
            .collect();
        (runs, tier, Some(transfer_id))
    })?;
    let Some((tier, ticket)) = posted else {
        return Ok(false);
    };
    directory.copy(group, record, tier, ticket);
    Ok(true)
}

/// Post a release's write-back batch at `TransferPriority::WriteBack`.
fn post_back(
    core: &mut Core,
    copier: &Copier,
    depth: usize,
    (group, record, _, tier): (usize, BankRecord, Staged, HostRecord),
) -> Result<bool, String> {
    let (directory, groups) = (&mut core.directory, &core.groups);
    let posted = copier.copies.post_within(depth, TransferPriority::WriteBack, || {
        let tensors = groups[group].tier.as_ref().expect("a write-back copies into its group's host tier");
        let runs = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| Run {
                source: groups[group].bank_view(index, record.index()),
                source_offset: 0,
                destination: tensor.records.tensor(u64::from(tier.0)).clone(),
                destination_offset: 0,
                bytes: tensor.records.bytes(),
            })
            .collect();
        (runs, ())
    })?;
    let Some(((), ticket)) = posted else {
        return Ok(false);
    };
    directory.back_copy(group, record, ticket);
    Ok(true)
}

/// Remaining permanent copy-engine promotion budget, excluding in-flight
/// opportunistic `Stage` batches so `stage_predicted` never starves permanent
/// VRAM promotions.
fn permanent_budget(
    max_depth: usize,
    total_flight: usize,
    staging_in_flight: usize,
    queued: usize,
    backs_queued: usize,
) -> usize {
    let permanent_in_flight = total_flight.saturating_sub(staging_in_flight);
    max_depth.saturating_sub(permanent_in_flight + queued + backs_queued)
}

/// `Copies::post_within` depth cap for a permanent batch (`WriteBack` or
/// `Promote`), excluding in-flight opportunistic `Stage` batches so already-
/// queued staging copies never lock out permanent write-backs or promotions.
fn permanent_post_depth(base_depth: usize, total_flight: usize, staging_in_flight: usize) -> usize {
    let permanent_in_flight = total_flight.saturating_sub(staging_in_flight);
    if permanent_in_flight >= base_depth {
        base_depth
    } else {
        base_depth + staging_in_flight
    }
}

/// The transfer queue's line a link move is priced and paced by.
fn line(core: &Core, shared: &Shared) -> Option<TierCopyCosts> {
    let copy = shared.copy.lock().expect("a copy's price").clone();
    assert!(
        copy.is_some() || !core.directory.link_moving(),
        "a link move runs before the transfer queue's line is measured"
    );
    copy
}

/// Each group's record bytes across all routed tensors (0 without a tier).
fn record_bytes(core: &Core) -> Vec<u64> {
    core.groups
        .iter()
        .map(|group| group.tier.as_ref().map_or(0, |tensors| tensors.iter().map(|tensor| tensor.records.bytes()).sum()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::LinkAdmitting;
    use crate::experts::directory::{Directory, HostTier, Placement};
    use crate::experts::horizon::Time;
    use crate::experts::interest::{Act, Class, Ledger, Name, Read as ReadState, Source, Stands, When};
    use crate::experts::map::{decode_stage_word, stage_word};
    use crate::experts::watcher::At;
    use magnitude_cost_model::{LinkIdleBudget, PathAdmission, PathKind, ReadLine, Terms};
    use std::cell::Cell;
    use std::time::Instant;

    /// Regression test for the `LinkMover::reserve()` VRAM collapse (`365a2e`):
    /// under high-miss MTP decode, permanent VRAM residency must NEVER be
    /// emptied into an idle reserve pool (`reserved(group) == 0`, keeping
    /// `>= capacity - 2` slots held while on-demand promotion releases at most
    /// `budget` victims).
    #[test]
    fn link_mover_never_collapses_permanent_vram_residency() {
        use crate::experts::directory::{HostRecord, Point, RoutedRead};
        use crate::experts::host_records::{Claim, Refuses, Taking};
        use crate::experts::reuse::{Evidence, Reuse};
        use crate::experts::watcher::Steps;

        let capacity = 240usize; // 4 layers * 60 slots/layer
        let layers = 4usize;
        let experts = 288usize;
        let mut directory = Directory::new(
            &[capacity],
            &vec![0; layers],
            &vec![experts; layers],
            HostTier {
                records: vec![(layers * experts) as u32],
                lent: vec![0],
                reserve: 0,
                placement: Placement::Replay,
            },
        );
        (0..layers).for_each(|layer| directory.price_copies(layer, Some(0.)));
        // Land all experts into the host tier.
        for layer in 0..layers {
            for expert in 0..experts {
                let Claim::Taken(grant) = directory.claim_landing(layer, expert, 0, Taking::Out, Refuses) else {
                    panic!("host tier has room for every expert");
                };
                directory.landed(layer, HostRecord(grant.record()), expert, None);
            }
        }
        let steps = Steps::new(
            crate::planning::FlightDepth::PIPELINED,
            1,
            crate::experts::exchange::Pulse::of(seismic::BackendName::Cpu),
        );
        let ranking = directory.ranking();
        // Seed ranking with high-miss MTP decode traffic so top experts are promoted.
        for seq in 1..=20u64 {
            {
                let time = directory.time();
                let mut times = time.lock();
                times.register(seq, crate::experts::horizon::Advance::Known(crate::experts::horizon::Tokens::new(4.0)));
            }
            for layer in 0..layers {
                let base = ((seq as usize) * 8) % experts;
                let mut chosen: Vec<usize> = (0..60).map(|i| (base + i) % experts).collect();
                chosen.sort_unstable();
                let read = RoutedRead { sequence: seq, layer, choices: chosen.len() as u64, experts: chosen };
                Reuse::of(&ranking).read(&read, None, Evidence::Decode);
            }
        }
        assert_eq!(directory.reserved(0), 0);

        // Fill all 240 permanent VRAM slots via promotions with `reserve == 0`.
        let mut ticket = 1u64;
        loop {
            let asked = steps.unbegun(|at| directory.promotions(at, 64));
            if asked.promotions.is_empty() {
                break;
            }
            let mut completed = Vec::new();
            for (group, record, (layer, expert)) in directory.queued() {
                let tier = directory.hold_tier(layer, expert).expect("tier kept");
                directory.copy(group, record, tier, ticket);
                ticket += 1;
                completed.push((group, record));
            }
            directory.promoted(completed, Point::whole(20, 0, 20), &steps);
        }
        let held_before: usize = (0..layers).map(|l| directory.held_count(l)).sum();
        assert_eq!(held_before, capacity, "all {capacity} permanent VRAM slots must be held");

        // Under continued high-miss traffic, `LinkMover` keeps `reserve == 0`
        // and asking `promotions(at, budget=2)` releases at most `budget` (2)
        // records on demand — NEVER collapsing permanent VRAM residency!
        for seq in 21..=40u64 {
            {
                let time = directory.time();
                let mut times = time.lock();
                times.register(seq, crate::experts::horizon::Advance::Known(crate::experts::horizon::Tokens::new(4.0)));
            }
            for layer in 0..layers {
                let base = ((seq as usize) * 16) % experts;
                let mut chosen: Vec<usize> = (0..32).map(|i| (base + i) % experts).collect();
                chosen.sort_unstable();
                let read = RoutedRead { sequence: seq, layer, choices: chosen.len() as u64, experts: chosen };
                Reuse::of(&ranking).read(&read, None, Evidence::Decode);
            }
        }
        let _mover = super::LinkMover::new();
        let asked = steps.unbegun(|at| directory.promotions(at, 2));
        let held_after: usize = (0..layers).map(|l| directory.held_count(l)).sum();
        assert_eq!(directory.reserved(0), 0, "LinkMover must keep promotion reserve at 0");
        assert!(asked.releases.len() <= 2, "on-demand releases must be bounded by budget=2, got {}", asked.releases.len());
        assert!(
            held_after >= capacity - 2,
            "permanent VRAM held ({held_after}) must remain >= capacity - 2 ({})",
            capacity - 2
        );
    }

    #[test]
    fn stage_word_falls_back_when_unlanded_or_stale_step() {
        assert_eq!(decode_stage_word(0, 1), None, "zero-initialized MAP_STAGE is unstaged");
        assert_eq!(decode_stage_word(0, 0), None, "step 0 with zero MAP_STAGE is unstaged");

        let word = stage_word(42, 0);
        assert_ne!(word, 0, "slot 0 must encode to a non-zero word");
        assert_eq!(decode_stage_word(word, 42), Some(0));
        assert_eq!(decode_stage_word(word, 41), None, "earlier step must fall back");
        assert_eq!(decode_stage_word(word, 43), None, "later step must fall back");

        let high_step = (1u64 << 16) + 7;
        let wrapped = stage_word(high_step, 127);
        assert_eq!(decode_stage_word(wrapped, high_step), Some(127));
        assert_eq!(decode_stage_word(wrapped, high_step + 1), None);
    }

    /// `NameLedger::schedule_for_path(PathKind::Link, ...)` schedules predicted
    /// experts in layer-ascending order for Link staging, regardless of whether
    /// the host tier is complete (100%-RAM) or unsettled (3-tier), and consumes
    /// `LinkIdleBudget` (including `DraftWindow`) across admitted moves.
    #[test]
    fn name_ledger_feeds_link_staging_in_layer_order_with_idle_budget() {
        let time = Time::new(&[vec![4, 8]], 16);
        let stands = Stands::new(time.clone());
        let mut ledger = Ledger::new(vec![0, 0, 0], 4, time, stands);
        let now = Instant::now();
        // Publish predicted hints for layers 2, 0, 1 for step 5.
        for (layer, expert) in [(2usize, 7usize), (0, 3), (1, 5)] {
            let name = Name {
                source: Source::Lead { distance: 1, row: None },
                when: When::Step(5),
                rank: 0,
                from: At::whole(5, layer.saturating_sub(1)),
                candidate: false, // e.g. tier_complete() == true (Located::Kept)
                lineage: None,
            };
            ledger.publish(layer, expert, name, now);
        }
        assert!(ledger.is_wanted(0, 3) && ledger.is_wanted(1, 5) && ledger.is_wanted(2, 7));

        let bytes = 2_000_000u64;
        let pcie_line = ReadLine::new(5e-6, 40e-6 / bytes as f64).unwrap();
        let admission = PathAdmission::link(
            Terms { compute: 0.020, drive: 0.0, link: 0.045, cpu: 0.025, memory: 0.015 },
            Some(pcie_line),
            Some(80e-6),
            None,
        );
        let idle = Cell::new(LinkIdleBudget::new(0.0, 100e-6, 0.0));
        let scheduled = ledger.schedule_for_path(
            PathKind::Link,
            ledger.wanted(),
            |_, _| ReadState::Absent,
            |_| LinkAdmitting { admission, bytes, idle: &idle },
        );
        let issued_layers: Vec<(usize, usize)> = scheduled
            .iter()
            .filter_map(|s| matches!(s.act, Act::Issue(Class::Expected { step: 5, .. })).then_some((s.layer, s.expert)))
            .collect();
        assert_eq!(issued_layers, vec![(0, 3), (1, 5), (2, 7)], "Link path orders staging layer-ascending");
        assert_eq!(idle.get().remaining(), 0.0, "idle budget consumed as moves were admitted");
    }

    /// Opportunistic `Stage` copies queued up to `ring_half >> copy.depth` must
    /// NEVER zero out `LinkMover::ask()`'s permanent promotion budget or lock
    /// out `post_promote()` / `post_back()` in `Copies::post_within()`.
    #[test]
    fn opportunistic_staging_flight_never_starves_permanent_promotions() {
        use super::{permanent_budget, permanent_post_depth};

        let base_depth = 2usize;
        let staging_in_flight = 8usize;

        // With 8 opportunistic `Stage` batches in flight and 0 permanent
        // batches, `ask()` still gets the full `base_depth = 2` budget.
        assert_eq!(permanent_budget(base_depth, staging_in_flight, staging_in_flight, 0, 0), 2);
        // `post_promote()` / `post_back()` receives depth `2 + 8 = 10 > 8`,
        // admitting up to 2 permanent batches even while 8 staging batches are
        // queued in `Copies`.
        assert_eq!(permanent_post_depth(base_depth, staging_in_flight, staging_in_flight), 10);

        // Once 1 permanent promotion is posted (`total_flight = 9`, `staging = 8`),
        // 1 permanent slot remains.
        assert_eq!(permanent_budget(base_depth, 9, staging_in_flight, 0, 0), 1);
        assert_eq!(permanent_post_depth(base_depth, 9, staging_in_flight), 10);

        // Once 2 permanent promotions are posted (`total_flight = 10`, `staging = 8`),
        // permanent budget is 0 and `permanent_post_depth` returns `base_depth = 2`
        // (`<= total_flight`), cleanly capping permanent flight at `base_depth`.
        assert_eq!(permanent_budget(base_depth, 10, staging_in_flight, 0, 0), 0);
        assert_eq!(permanent_post_depth(base_depth, 10, staging_in_flight), base_depth);

        // Queued promotions / write-backs still count against `ask()`'s budget.
        assert_eq!(permanent_budget(base_depth, staging_in_flight, staging_in_flight, 1, 1), 0);
    }
}
