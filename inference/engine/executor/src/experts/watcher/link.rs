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
use crate::experts::horizon::{Hold, Tokens};
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
pub(in crate::experts) struct Copier {
    pub copies: crate::experts::copies::Copies,
}

/// What a reserve is sized within ([`reserve`]): each layer's group and
/// the most experts a decode step reads there (its experts, at most a
/// step's choices), each group's capacity (`Directory::capacity`),
/// and the steps in flight.
struct Sizing {
    groups: Vec<usize>,
    reads: Vec<usize>,
    records: Vec<usize>,
    flight: usize,
}

/// A candidate reserve of a group ([`reserve`]): its records, and what it
/// saves a step less what it costs, in misses.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Reserve {
    records: usize,
    net: f64,
}

/// Each group's reserve, where what it saves exceeds what it costs most,
/// from the decode steps' depth curve `decoded` (every decode step read
/// since the load) and `drain`, the mean steps a decode release waited.
fn reserve(sizing: &Sizing, decoded: &MissCurve, drain: Option<f64>) -> Vec<Vec<Reserve>> {
    let groups = sizing.records.len();
    let (Some(drain), true) = (drain, decoded.steps > 0) else {
        return (0..groups).map(|_| vec![Reserve { records: 0, net: 0. }]).collect();
    };
    assert!(drain <= sizing.flight as f64, "a release waited {drain} steps, beyond the {} in flight", sizing.flight);
    let steps = decoded.steps as f64;
    (0..groups)
        .map(|group| {
            let members = (0..sizing.groups.len()).filter(|&layer| sizing.groups[layer] == group).collect::<Vec<_>>();
            let whole = sizing.records[group] / members.len().max(1);
            let left = sizing.records[group] - whole * members.len();
            let misses = |layer: usize, slots: usize| decoded.layers[layer].at(slots).1 as f64 / steps;
            for &layer in &members {
                let most = sizing.reads[layer] as f64;
                assert!(misses(layer, 0) <= most, "{} reads a step of a layer reading at most {most}", misses(layer, 0));
            }
            let asked = members.iter().map(|&layer| misses(layer, whole)).sum::<f64>() * drain;
            let rate = members.iter().map(|&layer| misses(layer, 0) - misses(layer, 1)).sum::<f64>() / members.len().max(1) as f64;
            (0..whole.max(1))
                .map(|levels| {
                    let records = left + levels * members.len();
                    let cost = members.iter().map(|&layer| misses(layer, whole - levels) - misses(layer, whole)).sum::<f64>();
                    Reserve { records, net: asked.min(records as f64) * rate - cost }
                })
                .collect()
        })
        .collect()
}

/// Of a group's reserves ([`reserve`]), the one to hold now, `current` held.
fn decide(candidates: &[Reserve], current: usize, hold: &Hold, now: Tokens, tokens_per_step: f64) -> usize {
    let best = candidates
        .iter()
        .copied()
        .max_by(|a, b| a.net.total_cmp(&b.net).then(b.records.cmp(&a.records)))
        .expect("a reserve's candidates");
    let Some(candidate) = candidates.iter().find(|candidate| candidate.records == current) else {
        return best.records;
    };
    let gain = (best.net - candidate.net) / tokens_per_step;
    let moved = best.records.abs_diff(current) as f64;
    match hold.pays(now, gain, moved) {
        true => best.records,
        false => current,
    }
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
/// `NameLedger`-driven opportunistic DMA staging, and `Reuse`-ranked permanent
/// promotions on `Copies`.
pub(super) struct LinkMover {
    drained: (u64, u64),
    holds: Vec<Hold>,
    /// Issue timestamps of in-flight opportunistic staging copies `(group, slot) -> (target_step, issued_at)`
    /// so `settle` can report landed link-staged hints to `NameLedger::link_landed`.
    stage_issued: HashMap<(usize, usize), (Step, Instant)>,
    /// Total opportunistic staging transfers landed (for one-time activation telemetry log).
    staged_total: u64,
}

impl LinkMover {
    pub(super) fn new() -> Self {
        Self { drained: (0, 0), holds: Vec::new(), stage_issued: HashMap::new(), staged_total: 0 }
    }

    /// Settle every completed write-back, permanent promotion, and opportunistic
    /// staging copy, and drain releases and landed staging slots past the
    /// frontier. Permanent moves report to `Telemetry::moved`; opportunistic
    /// staging reports separately to `Telemetry::staged_link`.
    pub(super) fn settle(&mut self, core: &mut Core, ledger: &mut Ledger, shared: &Shared, copier: &Copier, progress: Point) -> bool {
        let Some(copy) = line(core, shared) else {
            return false;
        };
        let mut moved = false;
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

        // 4. Drain releases and landed staging slots whose (target_step, layer)
        //    has been served at `progress`.
        moved |= core.directory.drain(progress, &core.steps);
        moved |= core.directory.drain_staging(progress, &core.steps);
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
        decoded: Option<&MissCurve>,
        held: bool,
        progress: Point,
    ) -> bool {
        if let Some(decoded) = decoded {
            self.size(core, decoded);
        }
        let Some(copy) = line(core, shared) else {
            return false;
        };
        let mut moved = self.settle(core, ledger, shared, copier, progress);
        if held || shared.life.ended().is_some() {
            return moved;
        }
        let bytes = record_bytes(core);

        // Priority 1: Write-backs (free bank records promotions/leases wait for).
        for back @ (group, record, ..) in core.directory.backs_queued() {
            match post_back(core, copier, copy.depth(bytes[group]), back) {
                Ok(true) => moved = true,
                Ok(false) => break,
                Err(error) => {
                    core.directory.written_back(group, record, false);
                    shared.end_serving(format!("a release's write-back into the host tier: {error}"));
                    return true;
                }
            }
        }

        // Priority 2: Opportunistic staging from NameLedger + DraftWindow carryover
        // (works on BOTH 100%-RAM and unsettled 3-tier workloads!).
        moved |= self.stage_predicted(core, ledger, shared, copier, &copy, &bytes, progress);

        // Priority 3: Permanent promotions filling remaining copy-engine depth.
        moved |= self.ask(core, shared, copier, &copy);
        let mut queued = core.directory.queued();
        queued.retain(|&(group, record, (layer, expert))| {
            let kept = core.directory.tier_record(layer, expert).is_some() && core.directory.reuse_now().in_top(layer, expert);
            if !kept {
                core.directory.unpromoted(group, record);
                moved = true;
            }
            kept
        });
        for promotion @ (group, record, _) in queued {
            match post_promote(core, copier, copy.depth(bytes[group]), promotion) {
                Ok(true) => moved = true,
                Ok(false) => break,
                Err(error) => {
                    core.directory.unpromoted(group, record);
                    shared.end_serving(format!("a promotion's copy into the bank: {error}"));
                    return true;
                }
            }
        }
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
        let mut moved = core.directory.drain_staging(progress, &core.steps);

        let admission = shared.telemetry.link_admission(Some(copy.transfer));
        let layers = core.layers.len();
        let c_layer = if layers == 0 { 0.0 } else { admission.terms.compute / layers as f64 };
        // Hints look up to `HINT_LEADS` layers ahead, so the compute overlap
        // horizon spans `HINT_LEADS * c_layer`.
        let c_window = (crate::planning::HINT_LEADS as f64) * c_layer;
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
                            match post_stage(core, copier, stage_depth, priority, group, slot, target_step, staged) {
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

            // DraftWindow carryover on Link: while an MTP draft window is active,
            // fill up to half of `next_step`'s parity ring with top Tier-1-only
            // Reuse carryover experts (sorted earliest-layer-first so layer 0..
            // warm up while keeping >= half the ring free for live within-pass
            // `routed_hint_route` hints once `next_step` begins).
            let carryover_cap = (ring_half / 2).max(1);
            let carryover_room = carryover_cap.saturating_sub(core.directory.staged_for_step(group, next_step));
            if draft_idle > 0.0 && carryover_room > 0 && copier.copies.flight() < stage_depth && idle.get().remaining() > 0.0 {
                let room = stage_depth.saturating_sub(copier.copies.flight()).min(carryover_room);
                let carryover = {
                    let reuse = core.directory.reuse_now();
                    reuse.top_candidates(room, |layer, expert| {
                        core.layers[layer].group == group
                            && core.directory.held(layer, expert).is_none()
                            && !core.directory.placing((layer, expert))
                            && core.directory.tier_record(layer, expert).is_some()
                            && core.directory.staged_entry(group, (layer, expert)).is_none()
                    })
                };
                let mut carryover: Vec<(f64, usize, usize)> = carryover
                    .into_iter()
                    .map(|(rank, layer, expert)| (rank.expected().uses(), layer, expert))
                    .collect();
                carryover.sort_by_key(|&(_, layer, _)| layer);
                for (uses, layer, expert) in carryover {
                    if copier.copies.flight() >= stage_depth {
                        break;
                    }
                    let mut budget = idle.get();
                    let t_bw = admission.transfer_seconds(record_bytes);
                    if !admission.admits(0.0, record_bytes, budget.remaining()) {
                        break;
                    }
                    let Some(slot) = core.directory.free_stage_slot(group, next_step) else {
                        break;
                    };
                    let priority = TransferPriority::Stage { step: next_step, layer, p: uses.clamp(0.0, 1.0) };
                    match post_stage(core, copier, stage_depth, priority, group, slot, next_step, (layer, expert)) {
                        Ok(Some((tier, ticket))) => {
                            self.stage_issued.insert((group, slot), (next_step, Instant::now()));
                            core.directory.stage(group, slot, (layer, expert), next_step, tier, ticket, (next_step & 1) as u8);
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

    /// Size each group's promotion reserve by `decoded` ([`reserve`], [`decide`]).
    fn size(&mut self, core: &mut Core, decoded: &MissCurve) {
        let choices = magnitude_kernels::constants::VERIFY_ROWS as usize * core.extents.selected;
        let sizing = Sizing {
            groups: core.directory.layer_groups(),
            reads: core.directory.layer_experts().into_iter().map(|experts| experts.min(choices)).collect(),
            records: core.directory.capacities(),
            flight: core.extents.words.flight as usize,
        };
        let drain = (self.drained.1 > 0).then(|| self.drained.0 as f64 / self.drained.1 as f64);
        let (now, spacing) = {
            let times = core.directory.time().lock();
            let now = times.clock.through_latest();
            let spacing = (0..sizing.records.len()).map(|group| times.tokens_per_step(group)).collect::<Vec<_>>();
            (now, spacing)
        };
        self.holds.resize(sizing.records.len(), Hold::started(now));
        for (group, candidates) in reserve(&sizing, decoded, drain).into_iter().enumerate() {
            let Some(tokens_per_step) = spacing[group] else {
                continue;
            };
            let current = core.directory.reserved(group);
            let records = decide(&candidates, current, &self.holds[group], now, tokens_per_step);
            if records != current {
                core.directory.reserve(group, records);
                self.holds[group].changed(now);
            }
        }
    }

    /// Ask the ranking for permanent promotions up to remaining copy-engine headroom.
    fn ask(&mut self, core: &mut Core, shared: &Shared, copier: &Copier, copy: &TierCopyCosts) -> bool {
        let max_depth = record_bytes(core)
            .into_iter()
            .filter(|&bytes| bytes > 0)
            .map(|bytes| copy.depth(bytes))
            .max()
            .unwrap_or(0);
        let pending = copier.copies.flight() + core.directory.queued().len() + core.directory.backs_queued().len();
        let budget = max_depth.saturating_sub(pending);
        if budget == 0 && !core.directory.has_surrenders() {
            return false;
        }
        if !core.directory.settle_changes() {
            return false;
        }
        let decoding = !core.steps.prompt_running();
        let (directory, drained) = (&mut core.directory, &mut self.drained);
        let asked = core.steps.unbegun_beside_frontier(|frontier, at| {
            let asked = directory.promotions(at, budget);
            if decoding {
                let releases = asked.releases.len() as u64;
                *drained = (drained.0 + releases * (at.next() - frontier), drained.1 + releases);
            }
            asked
        });
        shared.promoted.store(directory.asked(), std::sync::atomic::Ordering::Relaxed);
        !asked.promotions.is_empty() || !asked.releases.is_empty()
    }
}

/// Post one expert's opportunistic H2D staging copy runs followed by a
/// `TransferItem::Signal` writing `MAP_STAGE` on the copy engine itself.
#[allow(clippy::too_many_arguments)]
fn post_stage(
    core: &mut Core,
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
    let posted = copier.copies.post_items_within(depth, priority, || {
        let tier = directory.hold_tier(layer, expert).expect("a selected staging candidate's tier record is kept");
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
        (items, tier)
    })?;
    Ok(posted)
}

/// Post a permanent promotion's copy batch at `TransferPriority::Promote`.
fn post_promote(
    core: &mut Core,
    copier: &Copier,
    depth: usize,
    (group, record, (layer, expert)): (usize, BankRecord, Staged),
) -> Result<bool, String> {
    let (directory, groups) = (&mut core.directory, &core.groups);
    let posted = copier.copies.post_within(depth, TransferPriority::Promote, || {
        let tier = directory.hold_tier(layer, expert).expect("a queued promotion's tier record is kept");
        let tensors = groups[group].tier.as_ref().expect("a promotion copies from its group's host tier");
        let runs = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| Run {
                source: tensor.records.tensor(u64::from(tier.0)).clone(),
                source_offset: 0,
                destination: groups[group].bank_view(index, record.index()),
                destination_offset: 0,
                bytes: tensor.records.bytes(),
            })
            .collect();
        (runs, tier)
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

/// The transfer queue's line a link move is priced and paced by.
fn line(core: &Core, shared: &Shared) -> Option<TierCopyCosts> {
    let copy = shared.copy.lock().expect("a copy's price").clone();
    assert!(copy.is_some() || !core.directory.moving(), "a link move runs before the transfer queue's line is measured");
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
    use super::{LinkAdmitting, MissCurve, Reserve, Sizing, decide, reserve};
    use crate::experts::directory::LayerCurve;
    use crate::experts::horizon::{Hold, Time, Tokens};
    use crate::experts::interest::{Act, Class, Ledger, Name, Read as ReadState, Source, Stands, When};
    use crate::experts::map::{decode_stage_word, stage_word};
    use crate::experts::watcher::At;
    use magnitude_cost_model::{LinkIdleBudget, PathAdmission, PathKind, ReadLine, Terms};
    use std::cell::Cell;
    use std::time::Instant;

    fn sizing(records: usize) -> Sizing {
        Sizing { groups: vec![0], reads: vec![8], records: vec![records], flight: 3 }
    }

    fn curve(misses: &[u64]) -> MissCurve {
        MissCurve { steps: 100, layers: vec![LayerCurve { misses: misses.to_vec(), requests: vec![0; misses.len()], choices: 0 }] }
    }

    fn held(tokens: f64) -> (Hold, Tokens) {
        let at = Tokens::new(tokens);
        let mut hold = Hold::started(Tokens::ZERO);
        hold.changed(at);
        (hold, at)
    }

    #[test]
    fn a_reserve_is_priced_at_each_level_of_the_share() {
        let decoded = curve(&[100, 0, 0, 0, 100, 100, 0, 0, 100]);
        assert_eq!(reserve(&sizing(4), &decoded, None)[0], [Reserve { records: 0, net: 0. }]);
        let candidates = &reserve(&sizing(4), &decoded, Some(2.))[0];
        let records = candidates.iter().map(|candidate| candidate.records).collect::<Vec<_>>();
        assert_eq!(records, [0, 1, 2, 3]);
        let net = candidates.iter().map(|candidate| candidate.net).collect::<Vec<_>>();
        assert_eq!(net, [0., 1., 2., 3.]);
        let (hold, now) = held(10.);
        assert_eq!(decide(candidates, 0, &hold, now, 1.), 3, "repays moving 3 records within the 10 tokens expected held");
    }

    #[test]
    fn a_short_bank_missing_often_reserves_nothing() {
        let decoded = curve(&[100, 150, 350, 0, 0, 0, 0, 0, 200]);
        let candidates = &reserve(&sizing(2), &decoded, Some(1.))[0];
        let unbounded = Hold::started(Tokens::ZERO);
        assert_eq!(decide(candidates, 0, &unbounded, Tokens::ZERO, 1.), 0, "{candidates:?}");
    }

    #[test]
    fn the_most_a_load_can_measure_leaves_every_layer_a_record() {
        let decoded = curve(&[0, 0, 0, 0, 0, 0, 0, 0, 800]);
        let candidates = &reserve(&sizing(4), &decoded, Some(3.))[0];
        assert!(candidates.iter().all(|candidate| candidate.records <= 3), "{candidates:?}");
        assert_eq!(reserve(&sizing(1), &decoded, Some(3.))[0].iter().map(|candidate| candidate.records).collect::<Vec<_>>(), [0]);
    }

    #[test]
    fn a_reserve_changes_only_where_its_gain_repays_the_move() {
        let candidates = [Reserve { records: 0, net: 0. }, Reserve { records: 37, net: 0.5 }];
        let (ten, at_ten) = held(10.);
        let (seventy_four, at_seventy_four) = held(74.);
        let unbounded = Hold::started(Tokens::ZERO);
        assert_eq!(decide(&candidates, 0, &ten, at_ten, 1.), 0, "5 misses saved do not repay moving 37 records");
        assert_eq!(decide(&candidates, 0, &seventy_four, at_seventy_four, 1.), 37);
        assert_eq!(decide(&candidates, 0, &unbounded, Tokens::ZERO, 1.), 37, "before the first change, any positive gain pays");
        assert_eq!(decide(&candidates, 37, &unbounded, Tokens::ZERO, 1.), 37, "already the best");
        let even = [Reserve { records: 0, net: 0. }, Reserve { records: 4, net: 0. }];
        assert_eq!(decide(&even, 4, &unbounded, Tokens::ZERO, 1.), 4, "no gain moves nothing");
    }

    #[test]
    #[should_panic(expected = "of a layer reading at most 8")]
    fn a_rate_beyond_the_reads_panics() {
        reserve(&sizing(4), &curve(&[0, 0, 0, 0, 0, 0, 0, 0, 900]), Some(1.));
    }

    #[test]
    #[should_panic(expected = "beyond the 3 in flight")]
    fn a_drain_beyond_the_flight_panics() {
        reserve(&sizing(4), &curve(&[0, 0, 0, 0, 0, 0, 0, 0, 100]), Some(3.5));
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
}
