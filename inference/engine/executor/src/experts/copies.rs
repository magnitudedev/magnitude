//! The device's transfer queue (`seismic::Transfer`), the one owner of what
//! queues on it: the bank's promotions from a host tier, each its own
//! batch, posted in the order they are due while fewer than the queue's
//! depth are in flight (`TierCopyCosts::depth`, measured on the device):
//! the queue never idles between batches, and a copy a route waits for
//! waits behind at most that depth, never behind every promotion asked.
//! Nothing a step waits on queues here: every datum a device wait awaits is
//! read by device work already in the graph behind it, so a batch held in
//! the driver behind a parked step delays only a promotion. No promotion
//! copies into a lent bank record (`BankRecords::usable`), so a prompt
//! group's scratch, which a parked step waits on, never waits on a copy.
//!
//! An owner of serving posts a batch and never issues it: the queue's
//! issuer, a thread of its own, issues each in order ([`Copies::start`]).
//! An issue can wait: while host access to its tensors is held, and in the
//! driver behind the device's queue, which a parked step can fill. So no
//! owner of serving calls the device's queue, and none holds a lock another
//! needs while an issue waits. The queue's completions
//! are the load's events (`events`): its watch, a second thread, waits on
//! the queue's own completion of each batch issued (a semaphore wait where
//! the backend has one) and announces it, so no promotion looks at the
//! queue by time, and
//! [`Copies::completed`] reads only what the two threads learned.

use super::events::Events;
use super::transport::{ExpertTransport, ScratchGenId, TransferId};
use seismic::{Tensor, Transfer, TransferItem, TransferRun};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// The longest one wait of the watch on the queue lasts before it looks
/// whether the load stopped: a bound, not a latency, as the batch's
/// completion ends the wait.
const BOUND: Duration = Duration::from_millis(50);

/// Wait timeout used by `Copies::watch` when any running batch was issued with
/// `copy_synced` (which intentionally skips `cuLaunchHostFunc` wakeups so
/// compute waits via `cuStreamWaitEvent`). Keeps `running` draining promptly.
const SYNCED_BOUND: Duration = Duration::from_millis(1);

/// Maximum number of batches submitted concurrently to `Copies`'s hardware
/// transfer stream (double-buffered: 1 actively transferring + 1 queued right
/// behind it so the copy engine never bubbles). Keeping remaining batches in
/// `State::waiting` preserves strict `TransferPriority` preemption, `reprice`,
/// and `cancel`, prevents background promotions from flooding the H2D copy
/// engine against on-demand `DecodeDma`, and strictly bounds head-of-line
/// wait when `ExpertTransport::land_decode_misses` joins an in-flight hint.
const MAX_RUNNING: usize = 2;

/// One run of a batch (`seismic::TransferRun`): `bytes` from `source` at
/// `source_offset` to `destination` at `destination_offset`, held by the
/// queue until its batch completes. A transfer is ordered against no
/// device work, so the views it reads and writes stay held while it runs:
/// a pool moves no slab a view holds (`seismic::Pool::return_freed`), and
/// none it writes moves under it.
pub(in crate::experts) struct Run {
    pub source: Tensor,
    pub source_offset: u64,
    pub destination: Tensor,
    pub destination_offset: u64,
    pub bytes: u64,
}

/// One 32-bit signal written on the transfer stream (`TransferItem::Signal`)
/// once every run before it in the same batch has landed: used by the
/// opportunistic DMA staging ring to publish `MAP_STAGE` on the copy engine
/// with zero host round-trip.
pub(in crate::experts) struct Signal {
    pub word: Tensor,
    pub index: u64,
    pub value: i32,
}

/// One step of a [`Copies`] batch, in order: a DMA copy run or a stream signal.
pub(in crate::experts) enum BatchItem {
    Run(Run),
    Signal(Signal),
}

/// Dispatch priority for queued DMA batches on [`Copies`]:
/// `PrefillFill > WriteBack > Promote > Stage { step, layer, p }`.
///
/// Prefill scratch fills (`PrefillFill`) are on the critical path of prompt
/// passes and preempt all background traffic. Write-backs free bank records
/// that promotions and prompt leases wait for. Permanent promotions earn hits
/// across every future token and take priority over 1-step opportunistic
/// staging hints (`Stage`), which carry per-step/per-layer deadlines (soonest
/// step first, then earliest layer, then highest probability).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::experts) enum TransferPriority {
    PrefillFill,
    WriteBack,
    Promote,
    Stage { step: u64, layer: usize, p: f64 },
}

impl TransferPriority {
    /// Strict weak ordering where `Ordering::Less` means "dispatched earlier".
    fn cmp_dispatch(&self, other: &Self) -> std::cmp::Ordering {
        let rank = |prio: &Self| match prio {
            Self::PrefillFill => 0u8,
            Self::WriteBack => 1,
            Self::Stage { .. } => 2,
            Self::Promote => 3,
        };
        rank(self).cmp(&rank(other)).then_with(|| match (*self, *other) {
            (
                Self::Stage { step: ls, layer: ll, p: lp },
                Self::Stage { step: rs, layer: rl, p: rp },
            ) => ls.cmp(&rs).then_with(|| ll.cmp(&rl)).then_with(|| rp.total_cmp(&lp)),
            _ => std::cmp::Ordering::Equal,
        })
    }
}

/// Metadata attached to EVERY prefill scratch-fill batch (`PrefillFill`):
/// both intermediate streaming batches (`publish_sync = false`) and the final
/// batch (`publish_sync = true`) carry their `(layer, sequence, window,
/// generation)` so any intermediate refusal poisons the entire generation and
/// prevents later partial/final batches from publishing a broken sync.
#[derive(Clone, Copy)]
pub(in crate::experts) struct PrefillBatch {
    pub layer: usize,
    pub sequence: u64,
    pub window: u64,
    pub generation: ScratchGenId,
    pub publish_sync: bool,
}

struct QueuedBatch {
    ticket: u64,
    priority: TransferPriority,
    items: Vec<BatchItem>,
    stage_transfer: Option<TransferId>,
    prefill: Option<PrefillBatch>,
}

/// The device's transfer queue, shared by every holder of it: the batches
/// posted, issued and complete, and the load's events, which each batch
/// posted, issued and complete is one of.
#[derive(Clone)]
pub(in crate::experts) struct Copies {
    inner: Arc<Inner>,
}

struct Inner {
    transfer: Transfer,
    events: Arc<Events>,
    transport: Option<Arc<ExpertTransport>>,
    state: Mutex<State>,
}

/// In-flight batch metadata: ticket, completion value, held items, optional
/// tracked `ExpertTransport` transfer id, and whether it was issued synced.
type RunningBatch = (u64, u64, Vec<BatchItem>, Option<TransferId>, bool);

/// Where the queue's batches are, by ticket (consecutive from one; dispatched
/// in [`TransferPriority`] order, cancellable while waiting).
#[derive(Default)]
struct State {
    /// The last ticket posted (0: none).
    posted: u64,
    /// The batches posted and not yet issued, dispatched in priority order
    /// (`WriteBack > Stage { step, layer, p } > Promote`, FIFO on ties).
    waiting: Vec<QueuedBatch>,
    /// The batch being issued, and since when: an issue that waits shows
    /// here ([`Copies::describe`]).
    issuing: Option<(u64, std::time::Instant)>,
    /// Every ticket issued so far (for diagnostics).
    issued: u64,
    /// The batches issued and not yet complete: each ticket, the value its
    /// batch completes at, its items (held until then), any tracked
    /// `ExpertTransport` transfer id, and whether it was issued synced
    /// (`copy_synced`), in issue order (completion order on the single
    /// transfer stream).
    running: VecDeque<RunningBatch>,
    /// The batches the queue refused as issued, and why.
    refused: BTreeMap<u64, String>,
    /// Prefill generations poisoned by a refused partial/final batch: no later
    /// batch of a poisoned generation may be issued or publish a sync.
    poisoned_generations: std::collections::HashMap<ScratchGenId, String>,
    /// Cancelled batches' items deferred to drop on the issuer thread (outside
    /// `residency` and `NoBlockingAllocationGuard`, since dropping a final
    /// `SlabHold`/`Allocation` may touch pool or driver locks).
    cancelled: Vec<Vec<BatchItem>>,
    /// Why the queue faulted, once it has: no batch completes from then.
    fault: Option<String>,
}

impl State {
    /// The batches posted and not complete: waiting, being issued, or
    /// running.
    fn flight(&self) -> usize {
        self.waiting.len() + usize::from(self.issuing.is_some()) + self.running.len()
    }

    /// Pop the highest-priority waiting batch (ties broken by lowest ticket).
    fn pop_next(&mut self) -> Option<QueuedBatch> {
        let best = self
            .waiting
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.priority.cmp_dispatch(&b.priority).then_with(|| a.ticket.cmp(&b.ticket)))
            .map(|(idx, _)| idx)?;
        Some(self.waiting.swap_remove(best))
    }

    /// Update the priority of an unbegun waiting batch; returns `true` if
    /// found in `waiting`.
    fn reprice(&mut self, ticket: u64, priority: TransferPriority) -> bool {
        let Some(entry) = self.waiting.iter_mut().find(|entry| entry.ticket == ticket) else {
            return false;
        };
        entry.priority = priority;
        true
    }

    /// Cancel an unbegun waiting batch; returns `true` only if it was still in
    /// `waiting` and removed before `issue` popped it. Never cancels an
    /// issuing or running DMA batch. Defers dropping its items to the issuer
    /// thread so calling `cancel` inside `residency` / `NoBlockingAllocationGuard`
    /// can never drop a final `SlabHold`/`Allocation` under the lock.
    fn cancel(&mut self, ticket: u64) -> Option<Option<TransferId>> {
        let pos = self.waiting.iter().position(|entry| entry.ticket == ticket)?;
        let removed = self.waiting.swap_remove(pos);
        self.cancelled.push(removed.items);
        Some(removed.stage_transfer)
    }

    /// Whether `ticket` is still unbegun in `waiting`.
    fn is_waiting(&self, ticket: u64) -> bool {
        self.waiting.iter().any(|entry| entry.ticket == ticket)
    }

    /// Whether `ticket` is still in flight (waiting, issuing, or running).
    fn is_in_flight(&self, ticket: u64) -> bool {
        self.is_waiting(ticket)
            || self.issuing.is_some_and(|(t, _)| t == ticket)
            || self.running.iter().any(|&(t, _, _, _, _)| t == ticket)
    }
}

impl Copies {
    #[cfg(test)]
    pub fn new(transfer: Transfer, events: Arc<Events>) -> Self {
        Self { inner: Arc::new(Inner { transfer, events, transport: None, state: Mutex::default() }) }
    }

    pub fn with_transport(transfer: Transfer, events: Arc<Events>, transport: Arc<ExpertTransport>) -> Self {
        Self { inner: Arc::new(Inner { transfer, events, transport: Some(transport), state: Mutex::default() }) }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner.state.lock().expect("the transfer queue's batches")
    }

    /// Post `items` as a batch at `priority`: its ticket. The queue's issuer
    /// issues batches in [`TransferPriority`] order ([`Self::start`]); nothing
    /// that posts waits for the device's queue.
    fn post(
        &self,
        priority: TransferPriority,
        items: Vec<BatchItem>,
        stage_transfer: Option<TransferId>,
        prefill: Option<PrefillBatch>,
    ) -> u64 {
        let ticket = {
            let mut state = self.state();
            state.posted += 1;
            let ticket = state.posted;
            state.waiting.push(QueuedBatch { ticket, priority, items, stage_transfer, prefill });
            ticket
        };
        self.inner.events.notify();
        ticket
    }

    /// Post a prefill scratch-fill DMA batch at [`TransferPriority::PrefillFill`]
    /// (highest priority, never capped by background promotion depth). Every
    /// partial and final batch carries its [`PrefillBatch`] (including
    /// `generation`): when `publish_sync` is `true`, `Copies::issue` submits it
    /// via `copy_synced` and atomically publishes `(generation, TransferSync)`
    /// to `ExpertTransport`. Any refusal of an earlier partial batch poisons
    /// `generation` terminally so no later batch of that generation is issued
    /// or publishes a sync.
    pub fn post_prefill(&self, items: Vec<BatchItem>, prefill: PrefillBatch) -> Result<u64, String> {
        {
            let state = self.state();
            if let Some(fault) = &state.fault {
                return Err(fault.clone());
            }
            if let Some(reason) = state.poisoned_generations.get(&prefill.generation) {
                return Err(reason.clone());
            }
        }
        Ok(self.post(TransferPriority::PrefillFill, items, None, Some(prefill)))
    }

    /// Post the batch `form` forms at `priority`, only while fewer than `depth`
    /// batches are posted and not complete, formed only then: what `form`
    /// returned beside its runs, with its ticket; none while as many are.
    pub fn post_within<T>(
        &self,
        depth: usize,
        priority: TransferPriority,
        form: impl FnOnce() -> (Vec<Run>, T),
    ) -> Result<Option<(T, u64)>, String> {
        self.post_items_within(depth, priority, || {
            let (runs, formed) = form();
            (runs.into_iter().map(BatchItem::Run).collect(), formed)
        })
    }

    /// Post the mixed run/signal batch `form` forms at `priority`, only while
    /// fewer than `depth` batches are posted and not complete.
    pub fn post_items_within<T>(
        &self,
        depth: usize,
        priority: TransferPriority,
        form: impl FnOnce() -> (Vec<BatchItem>, T),
    ) -> Result<Option<(T, u64)>, String> {
        self.post_tracked_within(depth, priority, || {
            let (items, formed) = form();
            (items, formed, None)
        })
    }

    /// Post a mixed run/signal batch optionally tracked by `ExpertTransport`
    /// (`TransferId`), so `Copies::issue` records its `TransferSync` event and
    /// `Copies::watch` marks it landed.
    pub fn post_tracked_within<T>(
        &self,
        depth: usize,
        priority: TransferPriority,
        form: impl FnOnce() -> (Vec<BatchItem>, T, Option<TransferId>),
    ) -> Result<Option<(T, u64)>, String> {
        assert!(depth > 0, "a transfer queue keeps a batch in flight");
        {
            let state = self.state();
            if let Some(fault) = &state.fault {
                return Err(fault.clone());
            }
            if state.flight() >= depth {
                return Ok(None);
            }
        }
        let (items, formed, stage_transfer) = form();
        Ok(Some((formed, self.post(priority, items, stage_transfer, None))))
    }

    /// Reprice an unbegun queued batch; returns `true` if the batch was still
    /// waiting and its priority was updated.
    pub fn reprice(&self, ticket: u64, priority: TransferPriority) -> bool {
        self.state().reprice(ticket, priority)
    }

    /// Cancel an unbegun queued batch; returns `true` only if the batch had
    /// not yet been popped by the issuer. Never cancels an in-flight DMA.
    pub fn cancel(&self, ticket: u64) -> bool {
        let cancelled = self.state().cancel(ticket);
        if let Some(stage_transfer) = cancelled {
            if let (Some(transport), Some(id)) = (&self.inner.transport, stage_transfer) {
                transport.cancel_unbegun_hint(id);
            }
            self.inner.events.notify();
            true
        } else {
            false
        }
    }

    /// Whether `ticket` has already begun issuing (not waiting in the queue).
    pub fn is_begun(&self, ticket: u64) -> bool {
        !self.state().is_waiting(ticket)
    }

    /// The batches posted and not complete.
    pub fn flight(&self) -> usize {
        self.state().flight()
    }

    /// Whether no batch is posted and not complete.
    pub fn idle(&self) -> bool {
        self.flight() == 0
    }

    /// The batches posted so far.
    #[cfg(test)]
    pub fn posted(&self) -> u64 {
        self.state().posted
    }

    /// Whether the batch of `ticket` is complete; why it never will be (the
    /// queue refused it, or faulted). It reads what the queue's threads
    /// learned, and calls nothing of the device.
    pub fn completed(&self, ticket: u64) -> Result<bool, String> {
        let state = self.state();
        if let Some(reason) = state.refused.get(&ticket) {
            return Err(reason.clone());
        }
        if state.is_in_flight(ticket) {
            return match &state.fault {
                Some(fault) => Err(fault.clone()),
                None => Ok(false),
            };
        }
        Ok(true)
    }

    /// Where the queue's batches are, for the exchange's watchdog: posted,
    /// waiting to be issued, being issued and for how long, issued and
    /// running, refused, and the queue's fault.
    pub fn describe(&self) -> String {
        let state = self.state();
        format!(
            "posted through {}, {} waiting to be issued, issuing {:?}, issued through {}, running {:?}, refused {:?}, fault {:?}",
            state.posted,
            state.waiting.len(),
            state.issuing.map(|(ticket, since)| (ticket, since.elapsed())),
            state.issued,
            state.running.iter().map(|&(ticket, value, _, _, _)| (ticket, value)).collect::<Vec<_>>(),
            state.refused,
            state.fault,
        )
    }

    /// The queue's two threads until `stopped` holds, each waiting on the
    /// load's events and on the device alone, so no owner of serving ever
    /// does:
    ///
    /// - its issuer issues each batch posted, in order
    ///   ([`Transfer::copy`], which waits while host access to its tensors
    ///   is held, and can wait in the driver behind the device's queue);
    /// - its watch waits on the queue's own completion of the earliest
    ///   batch issued (a semaphore wait where the backend has one) and
    ///   announces it, or the queue's fault.
    ///
    /// Each issue, refusal, completion and fault is an event. `stopped` is
    /// looked at as each wait ends, the load's stop being an event too.
    pub fn start(&self, stopped: impl Fn() -> bool + Send + Sync + 'static) -> Result<Vec<JoinHandle<()>>, String> {
        let stopped = Arc::new(stopped);
        let issuer = {
            let (copies, stopped) = (self.clone(), stopped.clone());
            seismic::threads::spawn("expert-copies-issue", move || {
                seismic_io::waited_on();
                copies.issue(&*stopped)
            })
            .map_err(|error| format!("the transfer queue's issuer: {error}"))?
        };
        let copies = self.clone();
        let watch = seismic::threads::spawn("expert-copies", move || {
            seismic_io::waited_on();
            copies.watch(&*stopped)
        })
        .map_err(|error| format!("the transfer queue's watch: {error}"))?;
        Ok(vec![issuer, watch])
    }

    /// The issuer ([`Self::start`]).
    fn issue(&self, stopped: &dyn Fn() -> bool) {
        let events = &self.inner.events;
        loop {
            let seen = events.seen();
            if stopped() {
                return;
            }
            let (next, cancelled) = {
                let mut state = self.state();
                let cancelled = std::mem::take(&mut state.cancelled);
                let has_prefill = state.waiting.iter().any(|entry| entry.priority == TransferPriority::PrefillFill);
                let next = if has_prefill || state.running.len() < MAX_RUNNING {
                    state.pop_next()
                } else {
                    None
                };
                state.issuing = next.as_ref().map(|entry| (entry.ticket, std::time::Instant::now()));
                (next, cancelled)
            };
            drop(cancelled);
            let Some(QueuedBatch { ticket, items: batch, stage_transfer, prefill, .. }) = next else {
                events.wait(seen);
                continue;
            };
            // If an earlier partial batch for this prefill generation was
            // refused, fail this batch terminally before touching the transfer
            // stream or publishing any sync.
            if let Some(prefill) = prefill {
                let poisoned = {
                    let state = self.state();
                    state.poisoned_generations.get(&prefill.generation).cloned()
                };
                if let Some(reason) = poisoned {
                    if let Some(transport) = &self.inner.transport {
                        transport.poison_prefill_generation(prefill.generation, prefill.sequence, reason.clone());
                    }
                    let mut state = self.state();
                    state.issuing = None;
                    state.issued = ticket;
                    state.refused.insert(ticket, reason);
                    events.notify();
                    continue;
                }
            }
            let items = batch
                .iter()
                .map(|item| match item {
                    BatchItem::Run(run) => TransferItem::Run(TransferRun {
                        source: &run.source,
                        source_offset: run.source_offset,
                        destination: &run.destination,
                        destination_offset: run.destination_offset,
                        bytes: run.bytes,
                    }),
                    BatchItem::Signal(signal) => TransferItem::Signal {
                        word: &signal.word,
                        index: signal.index,
                        value: signal.value,
                    },
                })
                .collect::<Vec<_>>();
            let need_sync = stage_transfer.is_some() || prefill.is_some_and(|p| p.publish_sync);
            let issued = if need_sync {
                self.inner.transfer.copy_synced(&items).map(|(value, sync)| (value, Some(sync)))
            } else {
                self.inner.transfer.copy(&items).map(|value| (value, None))
            };
            drop(items);
            match issued {
                Ok((value, sync)) => {
                    if let (Some(transport), Some(id), Some(sync)) = (&self.inner.transport, stage_transfer, sync.as_ref()) {
                        transport.attach_sync(id, sync.clone());
                        transport.drain_deferred_syncs();
                    }
                    if let (Some(transport), Some(prefill), Some(sync)) =
                        (&self.inner.transport, prefill.filter(|p| p.publish_sync), sync)
                    {
                        transport.publish_prefill_sync(
                            prefill.layer,
                            prefill.sequence,
                            prefill.window,
                            prefill.generation,
                            Some(sync),
                        );
                    }
                    let mut state = self.state();
                    state.issuing = None;
                    state.issued = ticket;
                    state.running.push_back((ticket, value, batch, stage_transfer, need_sync));
                }
                Err(error) => {
                    let msg = error.to_string();
                    if let (Some(transport), Some(id)) = (&self.inner.transport, stage_transfer) {
                        transport.fault_transfer(id, msg.clone());
                        transport.drain_deferred_syncs();
                    }
                    if let Some(prefill) = prefill {
                        if let Some(transport) = &self.inner.transport {
                            transport.poison_prefill_generation(prefill.generation, prefill.sequence, msg.clone());
                        }
                    }
                    let mut state = self.state();
                    if let Some(prefill) = prefill {
                        state.poisoned_generations.entry(prefill.generation).or_insert_with(|| msg.clone());
                    }
                    state.issuing = None;
                    state.issued = ticket;
                    state.refused.insert(ticket, msg);
                }
            }
            events.notify();
        }
    }

    /// The watch ([`Self::start`]).
    fn watch(&self, stopped: &dyn Fn() -> bool) {
        let events = &self.inner.events;
        loop {
            let seen = events.seen();
            if stopped() {
                return;
            }
            let earliest = {
                let state = self.state();
                state
                    .fault
                    .is_none()
                    .then(|| state.running.front().map(|&(_, value, _, _, synced)| (value, synced)))
                    .flatten()
            };
            let Some((value, synced)) = earliest else {
                events.wait(seen);
                continue;
            };
            let timeout = if synced { SYNCED_BOUND } else { BOUND };
            match self.inner.transfer.wait(value, timeout) {
                Ok(false) => continue,
                Ok(true) => {
                    // Mark `Landed` in `ExpertTransport` BEFORE popping from
                    // `running` so `Copies::completed(batch) == Ok(true)`
                    // strictly implies `TransferState::Landed` when
                    // `LinkMover::settle` calls `try_release_hint_slot`.
                    let stage_transfer = self.state().running.front().and_then(|&(_, _, _, id, _)| id);
                    if let (Some(id), Some(transport)) = (stage_transfer, &self.inner.transport) {
                        transport.complete_dma(id);
                    }
                    let completed = self.state().running.pop_front();
                    if let Some((_, _, items, _, _)) = completed {
                        drop(items);
                    }
                    if let (Some(_), Some(transport)) = (stage_transfer, &self.inner.transport) {
                        transport.drain_deferred_syncs();
                    }
                }
                Err(error) => {
                    let msg = error.to_string();
                    let faulted_ids: Vec<TransferId> = {
                        let mut state = self.state();
                        state.fault = Some(msg.clone());
                        state.running.iter().filter_map(|&(_, _, _, id, _)| id).collect()
                    };
                    if let Some(transport) = &self.inner.transport {
                        for id in faulted_ids {
                            transport.fault_transfer(id, msg.clone());
                        }
                        transport.drain_deferred_syncs();
                    }
                }
            }
            events.notify();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(state: &mut State, priority: TransferPriority) -> u64 {
        state.posted += 1;
        let ticket = state.posted;
        state.waiting.push(QueuedBatch { ticket, priority, items: Vec::new(), stage_transfer: None, prefill: None });
        ticket
    }

    /// Priority ordering (`PrefillFill > WriteBack > Stage { step, layer, p } > Promote`),
    /// `reprice`, and `cancel` (only unbegun queued batches; never issuing/running).
    #[test]
    fn priority_preemption_reprice_and_cancel_never_cancels_in_flight() {
        let mut state = State::default();
        let t_promote = push(&mut state, TransferPriority::Promote);
        let t_stage_late = push(&mut state, TransferPriority::Stage { step: 5, layer: 2, p: 0.9 });
        let t_stage_l1_low = push(&mut state, TransferPriority::Stage { step: 4, layer: 1, p: 0.4 });
        let t_stage_l1_high = push(&mut state, TransferPriority::Stage { step: 4, layer: 1, p: 0.85 });
        let t_stage_l0 = push(&mut state, TransferPriority::Stage { step: 4, layer: 0, p: 0.3 });
        let t_wb = push(&mut state, TransferPriority::WriteBack);
        let t_prefill = push(&mut state, TransferPriority::PrefillFill);

        // PrefillFill preempts WriteBack, Promote, and all Stage batches.
        assert_eq!(state.pop_next().unwrap().ticket, t_prefill);

        // WriteBack preempts all Stage and Promote batches.
        let first = state.pop_next().unwrap();
        assert_eq!(first.ticket, t_wb);
        // Simulate `first` now issuing -> cancel must return None!
        state.issuing = Some((first.ticket, std::time::Instant::now()));
        assert!(state.cancel(t_wb).is_none(), "cannot cancel a batch already being issued");
        assert!(state.is_in_flight(t_wb));
        state.issuing = None;
        state.running.push_back((t_wb, 1, Vec::new(), None, false));
        assert!(state.cancel(t_wb).is_none(), "cannot cancel a running DMA batch");
        assert!(state.is_in_flight(t_wb));

        // Reprice `t_stage_late` to step 4, layer 0, p = 0.95 -> jumps ahead of `t_stage_l0` and `t_promote`.
        assert!(state.reprice(t_stage_late, TransferPriority::Stage { step: 4, layer: 0, p: 0.95 }));
        assert_eq!(state.pop_next().unwrap().ticket, t_stage_late);
        assert_eq!(state.pop_next().unwrap().ticket, t_stage_l0);

        // Cancel `t_stage_l1_low` while still unbegun -> succeeds, removes it
        // from flight, and defers its batch items to `cancelled` for drop on
        // the issuer thread rather than dropping under the caller's lock.
        assert!(state.cancel(t_stage_l1_low).is_some());
        assert!(!state.is_in_flight(t_stage_l1_low));
        assert_eq!(state.cancelled.len(), 1, "cancelled batch items are deferred for drop on the issuer thread");
        assert!(state.cancel(t_stage_l1_low).is_none(), "second cancel returns None");
        assert_eq!(state.cancelled.len(), 1);

        // Remaining order: live `t_stage_l1_high` preempts background `t_promote`.
        assert_eq!(state.pop_next().unwrap().ticket, t_stage_l1_high);
        assert_eq!(state.pop_next().unwrap().ticket, t_promote);
        assert!(state.pop_next().is_none());
    }

    /// Proof cases #1, #2, and #4 end-to-end on `ExpertTransport` + `Copies`:
    /// - Delayed prefill fill publishes its `TransferSync` before consumer
    ///   `take_prefill_sync` returns.
    /// - Overlapping consumer leases prohibit slot reuse until the FINAL reader
    ///   retires.
    /// - Cancellation of an unbegun hint frees its slot, whereas cancellation
    ///   of an in-flight hint is refused and keeps the slot pinned until DMA
    ///   completes and all readers release.
    #[test]
    fn delayed_fill_overlapping_readers_and_cancellation_slot_reuse_safety() {
        use crate::experts::transport::{DestSlot, ExpertKey, ExpertTransport, SourceRef};

        let transport = ExpertTransport::new_registry_only();

        // 1. Delayed prefill fill before consumer read: consumer waits on
        //    `take_prefill_sync` until the producer thread publishes.
        let producer_transport = Arc::clone(&transport);
        let producer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(15));
            producer_transport.publish_prefill_sync(2, 10, 0, None);
        });
        let got = transport.take_prefill_sync(2, 10, 0).expect("delayed prefill sync");
        assert!(got.is_none());
        producer.join().unwrap();

        // 2. Overlapping readers prohibit slot reuse until final reader retires,
        //    and cancelling an unbegun hint vs an in-flight hint preserves slot
        //    reuse safety.
        let key_unbegun = ExpertKey { group: 0, layer: 1, expert: 3 };
        let slot_unbegun = DestSlot { group: 0, slot: 1 };
        let unbegun_id = transport.register_hint(key_unbegun, 11, slot_unbegun);
        assert!(!transport.can_reuse_slot(slot_unbegun), "slot pinned while hint queued");
        assert!(transport.cancel_unbegun_hint(unbegun_id), "unbegun queued hint cancels cleanly");
        assert!(transport.can_reuse_slot(slot_unbegun), "unbegun cancelled hint frees slot safely");

        let key_live = ExpertKey { group: 0, layer: 1, expert: 7 };
        let src_live = SourceRef { tier_record: 22, generation: 2 };
        let slot_live = DestSlot { group: 0, slot: 2 };
        let live_id = transport.register_in_flight_hint(key_live, src_live, slot_live, None);
        assert!(!transport.cancel_unbegun_hint(live_id), "in-flight DMA cannot be cancelled");
        let joined1 = transport.join_by_key(key_live, src_live).expect("reader 1 joins in-flight hint");
        let joined2 = transport.join_by_key(key_live, src_live).expect("reader 2 joins in-flight hint");
        assert!(!transport.can_reuse_slot(slot_live), "slot pinned while DMA + 2 readers active");

        // DMA finishes, but both readers still hold leases -> slot reuse prohibited.
        transport.complete_dma(live_id);
        assert!(!transport.try_release_hint_slot(slot_live), "hint release blocked with 2 active readers");
        assert!(!transport.can_reuse_slot(slot_live), "slot reuse prohibited with 2 active readers");

        // Reader 1 cancels/retires early -> reader 2 still active -> slot reuse STILL prohibited!
        joined1.lease.cancel();
        assert!(!transport.can_reuse_slot(slot_live), "slot reuse prohibited until FINAL reader retires");

        // Final reader retires -> slot reuse now permitted!
        joined2.lease.release();
        assert!(transport.can_reuse_slot(slot_live), "slot reusable once final reader retires");
        assert!(transport.try_release_hint_slot(slot_live), "hint slot releases cleanly once unpinned");
    }
}
