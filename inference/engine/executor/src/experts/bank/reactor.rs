//! Persistent zero-switch 3-tier decode DMA reactor (`DecodeReactor`).
//!
//! Serves ONLY residual `HOST_WORK` (`FILE_PENDING`, `CPU_SITED`, or `CE_DMA`
//! overflow when `N_host_misses >= N_ce_min`) for unified streamed decode
//! graphs (`RunStage::Whole` on CUDA `Placer::Tier`):
//!
//! 1. On the fast path (`VRAM_HIT` + `HOST_STAGE` only, `flags == 0`),
//!    `routed_stage_host` self-signals `land_gate = g_L` on the GPU via
//!    `st.release.sys.global.u32` and never writes `ready_ring[slot].ready_gen`.
//!    `DecodeReactor` does zero host work and zero PCIe round-trips; the GPU's
//!    top-level `BatchMemOp(WRITE_VALUE_32: completed_gen = g_L)` advances the
//!    completion cursor past `g_L`.
//! 2. On the residual `HOST_WORK` path (`flags != 0`), `routed_stage_host`
//!    publishes `ready_ring[slot].ready_gen = g_L` (`st.release.sys.global.u64`).
//!    `DecodeReactor` observes `ready_gen == g_L`, retires any prior
//!    `DecodeLanding`s whose `gen <= completed_gen`, calls
//!    `ExpertBank::land_decode_misses_unified(...)` (preserving SM-staged
//!    `back_sources` entries), and signals `land_gate = g_L` via the Copy
//!    stream (`cuStreamWriteValue32` + WDDM flush) or host write.
//! 3. On any error, `DecodeReactor` ALWAYS signals `land_gate = g_L` so the
//!    GPU compute stream can never deadlock on `BatchMemOp(WAIT_VALUE_32)`,
//!    and surfaces the error in `drain_through` / `drain`.

use super::ExpertBank;
use crate::experts::transport::{DECODE_SOURCE_SLOTS, DecodeLanding, ExpertTransport};
use magnitude_kernels::constants::{
    READY_RING_FLAGS, READY_RING_GATE_ADDR, READY_RING_GEN, READY_RING_LAYER,
    READY_RING_MASK_HITS, READY_RING_READY_GEN, READY_RING_ROWS, READY_RING_SEQUENCE,
    READY_RING_SLOT, READY_RING_STAGE_BASE,
};
use seismic::{Device, Tensor, Timeline};
use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

/// Bound per-flight tensors and timeline handles for one unified streamed
/// decode block (`RunStage::Whole` on CUDA `Placer::Tier`).
pub(crate) struct UnifiedDecodeFlight {
    pub front_sources: Tensor,
    pub back_sources: Tensor,
    pub ready_slot: Tensor,
    pub land_gate: Timeline<seismic::timeline::Gate>,
    pub completed_gen: Timeline<seismic::timeline::Reached>,
    pub gen: u64,
}

struct ReactorState {
    /// Highest generation `g` that the reactor has either serviced or observed
    /// completed (self-gated) on the GPU.
    settled_gen: u64,
    /// Reserved flight generations cancelled before GPU submission committed
    /// (e.g. bind/attach/flush failure), skipped immediately if `reactor_loop`
    /// reaches them so refusal drain never hangs on an unsubmitted GPU gate.
    cancelled_gens: BTreeSet<u64>,
    /// Landings from `HOST_WORK` flights whose `DecodeBack(g)` is still in
    /// flight on GPU, retired in FIFO order as `completed_gen` advances.
    pending_landings: VecDeque<(u64, DecodeLanding)>,
    error: Option<String>,
    closed: bool,
}

struct ReactorShared {
    submitted_gen: AtomicU64,
    /// Diagnostic phase of `reactor_loop`:
    /// 0=init, 1=parked_idle, 2=polling_ready, 3=in_land_decode_misses,
    /// 4=in_signal_decode_gate, 5=retiring_completed.
    phase: std::sync::atomic::AtomicU32,
    cursor_gen: AtomicU64,
    bank: OnceLock<Weak<ExpertBank>>,
    transport: Arc<ExpertTransport>,
    state: Mutex<ReactorState>,
    cv: Condvar,
}

pub(crate) struct DecodeReactor {
    land_gate: Timeline<seismic::timeline::Gate>,
    land_gate_dev_addr: u64,
    completed_gen: Timeline<seismic::timeline::Reached>,
    next_gen: AtomicU64,
    slot_last_gen: [AtomicU64; DECODE_SOURCE_SLOTS],
    shared: Arc<ReactorShared>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl DecodeReactor {
    /// Start a persistent zero-switch decode DMA reactor for `device`.
    pub(crate) fn start(device: &Device, transport: Arc<ExpertTransport>) -> Result<Self, String> {
        let land_gate = Timeline::<seismic::timeline::Gate>::new(device)
            .map_err(|err| format!("failed to create decode land gate timeline: {err}"))?;
        let land_gate_dev_addr = land_gate.device_address().ok_or_else(|| {
            "decode land gate timeline has no device address on CUDA backend".to_string()
        })?;
        let completed_gen = Timeline::<seismic::timeline::Reached>::new(device)
            .map_err(|err| format!("failed to create decode completed_gen timeline: {err}"))?;
        let shared = Arc::new(ReactorShared {
            submitted_gen: AtomicU64::new(0),
            phase: std::sync::atomic::AtomicU32::new(0),
            cursor_gen: AtomicU64::new(1),
            bank: OnceLock::new(),
            transport,
            state: Mutex::new(ReactorState {
                settled_gen: 0,
                cancelled_gens: BTreeSet::new(),
                pending_landings: VecDeque::new(),
                error: None,
                closed: false,
            }),
            cv: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let worker_gate = land_gate.clone();
        let worker_completed = completed_gen.clone();
        let worker = seismic::threads::spawn("decode-dma-reactor", move || {
            reactor_loop(worker_shared, worker_gate, worker_completed);
        })
        .map_err(|err| format!("failed to spawn decode-dma-reactor thread: {err}"))?;
        Ok(Self {
            land_gate,
            land_gate_dev_addr,
            completed_gen,
            next_gen: AtomicU64::new(1),
            slot_last_gen: std::array::from_fn(|_| AtomicU64::new(0)),
            shared,
            worker: Some(worker),
        })
    }

    /// Attach the owning [`ExpertBank`] weak reference once `Arc::new(bank)`
    /// is constructed in `ExpertBank::open`.
    pub(crate) fn attach_bank(&self, bank: Weak<ExpertBank>) {
        let _ = self.shared.bank.set(bank);
    }

    /// Lock-free preparation of one unified streamed decode flight:
    /// - Allocates monotonic generation `g_L` and ring slot `(g_L - 1) % 8`.
    /// - Waits lock-free until `completed_gen` has reached the slot's previous
    ///   generation (almost always already reached with 8 slots).
    /// - Populates the host-mapped `ready_slot` descriptor header before
    ///   `cuGraphLaunch`.
    /// - Publishes `submitted_gen = g_L` and wakes the reactor if parked.
    pub(crate) fn prepare_flight(
        &self,
        group: usize,
        layer: usize,
        sequence: u64,
        rows: u64,
        mask_hits_back: bool,
    ) -> Result<UnifiedDecodeFlight, String> {
        let gen = self.next_gen.fetch_add(1, Ordering::Relaxed);
        let slot = ((gen - 1) as usize) % DECODE_SOURCE_SLOTS;

        // Wait lock-free until the GPU's top-level `WRITE_VALUE_32(completed_gen)`
        // proves the previous flight using `slot` has finished `DecodeBack`.
        let prev_gen = self.slot_last_gen[slot].load(Ordering::Acquire);
        if prev_gen > 0 {
            let started = Instant::now();
            let mut warned = false;
            let mut spins = 0u32;
            while !self.completed_gen.reached(prev_gen).unwrap_or(true) {
                if self.shared.transport.is_ended() {
                    return Err("serving ended while waiting for decode ring slot reuse".to_string());
                }
                if spins < 256 {
                    std::hint::spin_loop();
                    spins += 1;
                } else {
                    if !warned && started.elapsed() >= Duration::from_secs(5) {
                        warned = true;
                        let ready_words = self.shared.transport.decode_ready_slot_words(slot).ok();
                        let ready_gen = ready_words.map(|w| w.load_u64(READY_RING_READY_GEN as usize));
                        let flags = ready_words.map(|w| w.load(READY_RING_FLAGS as usize));
                        let slot_layer = ready_words.map(|w| w.load(READY_RING_LAYER as usize));
                        let slot_rows = ready_words.map(|w| w.load(READY_RING_ROWS as usize));
                        let slot_stage_base = ready_words.map(|w| w.load(READY_RING_STAGE_BASE as usize));
                        let gate_signalled = self.land_gate.signalled(prev_gen).ok();
                        let (settled, err) = {
                            let st = self.shared.state.lock().expect("decode reactor state");
                            (st.settled_gen, st.error.clone())
                        };
                        tracing::warn!(
                            slot,
                            prev_gen,
                            current = self.completed_gen.current().unwrap_or(0),
                            gate_signalled = ?gate_signalled,
                            ready_gen = ?ready_gen,
                            flags = ?flags,
                            slot_layer = ?slot_layer,
                            slot_rows = ?slot_rows,
                            slot_stage_base = ?slot_stage_base,
                            settled,
                            reactor_phase = self.shared.phase.load(Ordering::Acquire),
                            reactor_cursor = self.shared.cursor_gen.load(Ordering::Acquire),
                            reactor_err = ?err,
                            elapsed_s = started.elapsed().as_secs_f64(),
                            "DecodeReactor::prepare_flight stalled >5s waiting for slot reuse"
                        );
                    }
                    std::thread::yield_now();
                }
            }
        }
        self.slot_last_gen[slot].store(gen, Ordering::Release);

        let stage_base = self.shared.transport.stage_base(group)?;
        let ready_words = self.shared.transport.decode_ready_slot_words(slot)?;
        ready_words.store_u64(READY_RING_READY_GEN as usize, 0);
        ready_words.store_u64(READY_RING_GEN as usize, gen);
        ready_words.store(READY_RING_LAYER as usize, layer as i32);
        ready_words.store(READY_RING_ROWS as usize, rows as i32);
        ready_words.store_u64(READY_RING_SEQUENCE as usize, sequence);
        ready_words.store(READY_RING_FLAGS as usize, 0);
        ready_words.store(READY_RING_SLOT as usize, slot as i32);
        ready_words.store_u64(READY_RING_GATE_ADDR as usize, self.land_gate_dev_addr);
        ready_words.store(READY_RING_STAGE_BASE as usize, stage_base as i32);
        ready_words.store(
            READY_RING_MASK_HITS as usize,
            if mask_hits_back { 1 } else { 0 },
        );
        std::sync::atomic::fence(Ordering::Release);

        let (front_sources, back_sources, ready_slot) =
            self.shared.transport.decode_slot_views(slot, rows)?;

        // Do NOT publish `submitted_gen` here: the caller commits via
        // `commit_flight(gen)` only after the GPU submission succeeds, or calls
        // `cancel_flight(gen)` on bind/attach/submit refusal so `drain()` can
        // never hang waiting on an unsubmitted GPU gate.
        Ok(UnifiedDecodeFlight {
            front_sources,
            back_sources,
            ready_slot,
            land_gate: self.land_gate.clone(),
            completed_gen: self.completed_gen.clone(),
            gen,
        })
    }

    /// Commit a reserved flight after its GPU submission succeeded: publishes
    /// `submitted_gen` and wakes `reactor_loop`.
    pub(crate) fn commit_flight(&self, gen: u64) {
        self.shared.submitted_gen.fetch_max(gen, Ordering::Release);
        {
            let _guard = self.shared.state.lock().expect("decode reactor state");
            self.shared.cv.notify_all();
        }
    }

    /// Cancel a reserved flight that was never submitted to the GPU (bind,
    /// attach, park registration, or flush failed), releasing its slot
    /// reservation and recording `gen` so later commits skip it cleanly.
    pub(crate) fn cancel_flight(&self, gen: u64) {
        if gen == 0 {
            return;
        }
        let slot = ((gen - 1) as usize) % DECODE_SOURCE_SLOTS;
        let _ = self.slot_last_gen[slot].compare_exchange(
            gen,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let mut state = self.shared.state.lock().expect("decode reactor state");
        state.cancelled_gens.insert(gen);
        self.shared.cv.notify_all();
    }

    /// Wait until all unified decode flights through `target_gen` have settled
    /// in the reactor, retire any `pending_landings` with `gen <= target_gen`,
    /// and return any reactor error.
    pub(crate) fn drain_through(&self, target_gen: u64) -> Result<(), String> {
        let mut state = self.shared.state.lock().expect("decode reactor state");
        let started = Instant::now();
        let mut warned = false;
        while state.settled_gen < target_gen && !state.closed {
            let (guard, timeout) = self
                .shared
                .cv
                .wait_timeout(state, Duration::from_millis(5))
                .expect("decode reactor state");
            state = guard;
            if timeout.timed_out()
                && state.settled_gen < target_gen
                && !warned
                && started.elapsed() >= Duration::from_secs(5)
            {
                warned = true;
                tracing::warn!(
                    target_gen,
                    settled_gen = state.settled_gen,
                    completed = self.completed_gen.current().unwrap_or(0),
                    elapsed_s = started.elapsed().as_secs_f64(),
                    "DecodeReactor::drain_through stalled >5s"
                );
            }
        }
        let mut to_retire = Vec::new();
        while state
            .pending_landings
            .front()
            .is_some_and(|(gen, _)| *gen <= target_gen)
        {
            if let Some((_, landing)) = state.pending_landings.pop_front() {
                to_retire.push(landing);
            }
        }
        let err = state.error.take();
        drop(state);

        if let Some(bank) = self.shared.bank.get().and_then(Weak::upgrade) {
            for landing in to_retire {
                bank.complete_decode_landing(landing);
            }
        } else {
            for landing in to_retire {
                let _ = landing.release_transport(&self.shared.transport);
            }
        }
        match err {
            Some(msg) => Err(msg),
            None => Ok(()),
        }
    }

    /// Wait until all submitted unified decode flights have settled, retire
    /// all pending landings, and return any reactor error.
    pub(crate) fn drain(&self) -> Result<(), String> {
        let target = self.shared.submitted_gen.load(Ordering::Acquire);
        if target == 0 {
            return Ok(());
        }
        self.drain_through(target)
    }
}

impl Drop for DecodeReactor {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().expect("decode reactor state");
            state.closed = true;
            self.shared.cv.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let remaining: Vec<DecodeLanding> = {
            let mut state = self.shared.state.lock().expect("decode reactor state");
            state
                .pending_landings
                .drain(..)
                .map(|(_, landing)| landing)
                .collect()
        };
        if let Some(bank) = self.shared.bank.get().and_then(Weak::upgrade) {
            for landing in remaining {
                bank.complete_decode_landing(landing);
            }
        } else {
            for landing in remaining {
                let _ = landing.release_transport(&self.shared.transport);
            }
        }
    }
}

fn reactor_loop(
    shared: Arc<ReactorShared>,
    land_gate: Timeline<seismic::timeline::Gate>,
    completed_gen: Timeline<seismic::timeline::Reached>,
) {
    let mut cursor: u64 = 1;
    let mut idle_spins: u32 = 0;

    loop {
        // 1. Check GPU completion cursor and retire any completed landings.
        let completed = completed_gen.current().unwrap_or(0);
        if completed > 0 {
            let mut retired = Vec::new();
            let mut cursor_advanced = false;
            {
                let mut state = shared.state.lock().expect("decode reactor state");
                while state
                    .pending_landings
                    .front()
                    .is_some_and(|(gen, _)| *gen <= completed)
                {
                    if let Some((_, landing)) = state.pending_landings.pop_front() {
                        retired.push(landing);
                    }
                }
                if completed >= cursor {
                    cursor = completed + 1;
                    state.settled_gen = state.settled_gen.max(completed);
                    state.cancelled_gens.retain(|&g| g > completed);
                    cursor_advanced = true;
                }
                if cursor_advanced || !retired.is_empty() {
                    shared.cv.notify_all();
                }
            }
            if !retired.is_empty() {
                shared.phase.store(5, Ordering::Release);
                if let Some(bank) = shared.bank.get().and_then(Weak::upgrade) {
                    for landing in retired {
                        bank.complete_decode_landing_deferred(landing);
                    }
                } else {
                    for landing in retired {
                        let _ = landing.release_transport(&shared.transport);
                    }
                }
            }
        }

        shared.cursor_gen.store(cursor, Ordering::Release);
        let submitted = shared.submitted_gen.load(Ordering::Acquire);
        if cursor > submitted {
            shared.phase.store(1, Ordering::Release);
            let state = shared.state.lock().expect("decode reactor state");
            if state.closed {
                break;
            }
            if cursor > shared.submitted_gen.load(Ordering::Acquire) {
                let timeout = if state.pending_landings.is_empty() {
                    Duration::from_millis(5)
                } else {
                    Duration::from_micros(200)
                };
                let _ = shared.cv.wait_timeout(state, timeout);
            }
            idle_spins = 0;
            continue;
        }
        // Skip any reserved generation cancelled before GPU submission committed.
        {
            let mut state = shared.state.lock().expect("decode reactor state");
            if state.cancelled_gens.remove(&cursor) {
                state.settled_gen = state.settled_gen.max(cursor);
                cursor += 1;
                shared.cv.notify_all();
                idle_spins = 0;
                continue;
            }
        }

        // 2. Poll `ready_ring[slot].ready_gen` (Acquire) for `cursor`.
        shared.phase.store(2, Ordering::Release);
        let slot = ((cursor - 1) as usize) % DECODE_SOURCE_SLOTS;
        let ready_words = match shared.transport.decode_ready_slot_words(slot) {
            Ok(w) => w,
            Err(_) => break,
        };
        let ready_gen = ready_words.load_u64(READY_RING_READY_GEN as usize);
        std::sync::atomic::fence(Ordering::Acquire);

        if ready_gen == cursor {
            idle_spins = 0;
            let wait_start = Instant::now();
            let layer = ready_words.load(READY_RING_LAYER as usize) as usize;
            let rows = ready_words.load(READY_RING_ROWS as usize) as u64;
            let sequence = ready_words.load_u64(READY_RING_SEQUENCE as usize);
            let mask_hits_back = ready_words.load(READY_RING_MASK_HITS as usize) != 0;

            // Harvest any completed prior landings to fold into this layer's residency lock.
            let done_now = completed_gen.current().unwrap_or(0);
            let prev_landings: Vec<DecodeLanding> = {
                let mut state = shared.state.lock().expect("decode reactor state");
                let mut list = Vec::new();
                while state
                    .pending_landings
                    .front()
                    .is_some_and(|(gen, _)| *gen <= done_now)
                {
                    if let Some((_, landing)) = state.pending_landings.pop_front() {
                        list.push(landing);
                    }
                }
                list
            };

            let outcome: Result<DecodeLanding, String> = (|| {
                let bank = shared
                    .bank
                    .get()
                    .and_then(Weak::upgrade)
                    .ok_or_else(|| "decode reactor bank dropped while flight active".to_string())?;
                shared.phase.store(3, Ordering::Release);
                let landing = bank.land_decode_misses_unified(
                    layer,
                    sequence,
                    rows,
                    prev_landings,
                    mask_hits_back,
                    slot,
                )?;
                shared.phase.store(4, Ordering::Release);
                let has_dma = !landing.syncs.is_empty();
                bank.transport()
                    .signal_decode_gate(&land_gate, cursor, has_dma)?;
                bank.record_dma_host_wait(wait_start.elapsed());
                Ok(landing)
            })();

            let mut state = shared.state.lock().expect("decode reactor state");
            match outcome {
                Ok(landing) => {
                    state.pending_landings.push_back((cursor, landing));
                }
                Err(err) => {
                    // Always raise `land_gate` on error so `DecodeBack` on the
                    // GPU compute stream never deadlocks on `WAIT_VALUE_32`.
                    let _ = land_gate.signal(cursor);
                    if state.error.is_none() {
                        state.error = Some(err);
                    }
                }
            }
            state.settled_gen = state.settled_gen.max(cursor);
            cursor += 1;
            shared.cv.notify_all();
        } else if land_gate.signalled(cursor).unwrap_or(false) {
            // Fast path: `routed_stage_host(cursor)` self-gated on GPU (`flags == 0`).
            idle_spins = 0;
            let mut state = shared.state.lock().expect("decode reactor state");
            state.settled_gen = state.settled_gen.max(cursor);
            cursor += 1;
            shared.cv.notify_all();
        } else {
            // Either `routed_stage_host(cursor)` has not finished yet, or it
            // self-gated (`flags == 0`) and `DecodeBack(cursor)` is running.
            if shared.state.lock().expect("decode reactor state").closed {
                break;
            }
            if idle_spins < 256 {
                std::hint::spin_loop();
                idle_spins += 1;
            } else if idle_spins < 512 {
                std::thread::yield_now();
                idle_spins += 1;
            } else {
                std::thread::sleep(Duration::from_micros(20));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// If a reserved unified decode flight is cancelled before GPU submission
    /// commits (e.g. bind/attach/flush refusal), `submitted_gen` is never
    /// published for that generation so `drain()` returns immediately without
    /// hanging, and a subsequent committed flight still settles cleanly.
    #[test]
    fn cancelled_unsubmitted_flight_never_hangs_drain_and_subsequent_commit_settles() {
        let device = crate::test_support::open_backend(
            &seismic::Devices::discover(&[]).unwrap(),
            seismic::BackendName::Cpu,
        )
        .unwrap();
        let land_gate = Timeline::<seismic::timeline::Gate>::new(&device).unwrap();
        let completed_gen = Timeline::<seismic::timeline::Reached>::new(&device).unwrap();
        let transport = ExpertTransport::new(None);
        let shared = Arc::new(ReactorShared {
            submitted_gen: AtomicU64::new(0),
            phase: std::sync::atomic::AtomicU32::new(0),
            cursor_gen: AtomicU64::new(1),
            bank: OnceLock::new(),
            transport,
            state: Mutex::new(ReactorState {
                settled_gen: 0,
                cancelled_gens: BTreeSet::new(),
                pending_landings: VecDeque::new(),
                error: None,
                closed: false,
            }),
            cv: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let worker_gate = land_gate.clone();
        let worker_completed = completed_gen.clone();
        let worker = seismic::threads::spawn("test-decode-reactor", move || {
            reactor_loop(worker_shared, worker_gate, worker_completed);
        })
        .unwrap();
        let reactor = DecodeReactor {
            land_gate: land_gate.clone(),
            land_gate_dev_addr: 0,
            completed_gen: completed_gen.clone(),
            next_gen: AtomicU64::new(1),
            slot_last_gen: std::array::from_fn(|_| AtomicU64::new(0)),
            shared,
            worker: Some(worker),
        };

        // Reserve gen 1, then cancel it (simulating bind/attach/flush failure).
        let gen1 = reactor.next_gen.fetch_add(1, Ordering::Relaxed);
        reactor.slot_last_gen[0].store(gen1, Ordering::Release);
        reactor.cancel_flight(gen1);
        assert_eq!(reactor.slot_last_gen[0].load(Ordering::Acquire), 0);
        assert_eq!(reactor.shared.submitted_gen.load(Ordering::Acquire), 0);
        // Refusal drain must return immediately without hanging.
        reactor.drain().unwrap();

        // Next flight (gen 2) commits and completes on GPU (self-gated via land_gate + completed_gen):
        // reactor must skip cancelled gen 1 and settle gen 2 cleanly.
        let gen2 = reactor.next_gen.fetch_add(1, Ordering::Relaxed);
        assert_eq!(gen2, 2);
        land_gate.signal(gen2).unwrap();
        completed_gen.reach(gen2).unwrap();
        reactor.commit_flight(gen2);
        reactor.drain_through(gen2).unwrap();
        reactor.drain().unwrap();
    }
}
