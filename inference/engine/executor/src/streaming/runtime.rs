//! The host side of a streamed load's steps (spec 4.3–4.9): the routed
//! layers' directory words, the staging pools, and the watcher thread that
//! serves requests, hints and scratch fills while steps run.
//!
//! The device side of each routed layer, per bank-served step:
//!
//! ```text
//! route, resolve, hints (layers l+1, l+2) → fill: park, convert the absent
//!       → resident gate/up and output over the bound slots
//! ```
//!
//! Every step's routed entries run in submission order on one queue, which
//! gives the watcher a clock: a hint ring of layer `l` carrying step
//! sequence `q` proves every entry of earlier layers of step `q` (and of
//! every earlier step) has run, including their converts.

use super::shared_directory::{Clock, SharedDirectory};
use super::source::{ExpertSource, Outcome, Priority, Range, Ticket};
use super::words::{SharedBytes, SharedWords};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Words of a layer's request region (`routed_route_streamed` in routed.seismic).
pub(crate) const REQUEST: usize = 0;
pub(crate) const ACKNOWLEDGED: usize = 1;
pub(crate) const FAILURE: usize = 2;
pub(crate) const COUNT: usize = 3;
pub(crate) const EXPERTS: usize = 4;

/// Words of a layer's scratch request region (`routed_scratch_request`):
/// the request and failure words as above; each round's acknowledgement and
/// arrival follow the header.
const SCRATCH_HEADER: usize = 4;

/// The routed tensors of an expert record, in convert order.
pub(crate) const KINDS: usize = 3;

/// How long a parked step may wait for its fetches (spec 4.6).
pub(crate) const PARK_BOUND: Duration = Duration::from_secs(1);

/// One routed tensor of a layer as the model file stores it: expert `e`'s
/// bytes are `[base + e * bytes, + bytes)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StoredTensor {
    pub base: u64,
    pub bytes: u64,
    /// The staging pool records of this tensor go to.
    pub pool: usize,
}

/// One staging pool: `records` stored records of one (kind, source element,
/// matrix shape), each `record_bytes` long, host-addressed.
pub(crate) struct StagingPool {
    pub bytes: SharedBytes,
    pub record_bytes: u64,
    /// Free records, longest free first: a free record keeps the stored
    /// bytes it last held until it is taken again.
    free: VecDeque<u32>,
    /// Per record, the (layer, expert) whose stored bytes a free record
    /// still holds.
    holds: Vec<Option<(usize, i32)>>,
    /// Records only a demand read takes: a launch's choices (spec 4.3).
    reserve: usize,
}

impl StagingPool {
    /// `records` of which `prefetch` may hold prefetched and admitted
    /// records; the rest stay for demand reads.
    pub(crate) fn new(bytes: SharedBytes, records: u32, prefetch: u32, record_bytes: u64) -> Self {
        Self {
            bytes,
            record_bytes,
            free: (0..records).collect(),
            holds: vec![None; records as usize],
            reserve: records.saturating_sub(prefetch) as usize,
        }
    }
}

/// A routed layer's device words and its host directory.
pub(crate) struct StreamedLayer {
    pub slot_map: SharedWords,
    /// Per slot of the layer's bank group, the step that last bound it.
    pub stamps: SharedWords,
    /// Per slot, the step whose conversion into it the device committed.
    pub commits: SharedWords,
    pub control: SharedWords,
    /// [5, P]: per absent choice of a request, its slot, its gate, up and
    /// down staging records (or records of -1 when the slot already holds
    /// the expert) and its acknowledgement (`routed_bank_fill`).
    pub placement: SharedWords,
    /// The layer's scratch request region (spec 4.9): every layer of a
    /// scratch-served step requests under the same sequence, so each has
    /// its own.
    pub scratch_control: SharedWords,
    /// Hints for this layer made one and two layers earlier.
    pub rings: [SharedWords; 2],
    pub stored: [StoredTensor; KINDS],
    /// Host copy: expert → slot, slot → expert.
    slot_of: Vec<i32>,
    expert_of: Vec<i32>,
    /// The last request sequence served and ring sequences read.
    served: i32,
    scratch_served: i32,
    read: [i32; 2],
}

impl StreamedLayer {
    pub(crate) fn new(
        slot_map: SharedWords,
        stamps: SharedWords,
        commits: SharedWords,
        control: SharedWords,
        placement: SharedWords,
        scratch_control: SharedWords,
        rings: [SharedWords; 2],
        stored: [StoredTensor; KINDS],
    ) -> Self {
        let experts = slot_map.len();
        let slots = stamps.len();
        // A bank starts empty (rule 5).
        slot_map.fill(-1);
        stamps.fill(-1);
        control.fill(0);
        placement.fill(0);
        scratch_control.fill(0);
        for ring in &rings {
            ring.fill(0);
        }
        Self {
            slot_map,
            stamps,
            commits,
            control,
            placement,
            scratch_control,
            rings,
            stored,
            slot_of: vec![-1; experts],
            expert_of: vec![-1; slots],
            served: 0,
            scratch_served: 0,
            read: [0; 2],
        }
    }
}

/// The host's heartbeat word: advanced while the watcher serves submitted
/// steps, including while it waits on reads, so a parked device wait tells a
/// slow host from a dead one (`metal/lib/streaming/control.h`).
pub(crate) struct Heartbeat {
    word: SharedWords,
    beat: i32,
    last: Instant,
}

/// How often the heartbeat advances: far inside a parked wait's patience,
/// rare enough to cost the memory system nothing.
const BEAT: Duration = Duration::from_micros(500);

impl Heartbeat {
    pub(crate) fn new(word: SharedWords) -> Self {
        word.fill(0);
        Self {
            word,
            beat: 0,
            last: Instant::now(),
        }
    }

    fn beat(&mut self) {
        if self.last.elapsed() >= BEAT {
            self.beat = self.beat.wrapping_add(1);
            self.word.store(0, self.beat);
            self.last = Instant::now();
        }
    }
}

/// How long the watcher rests after a pass that found nothing: a park
/// waits at most this long to be noticed, and the watcher does not spin on
/// memory the device is using.
const IDLE: Duration = Duration::from_micros(20);

/// Wait for `ticket` until `deadline`, beating meanwhile.
fn wait_ticket(ticket: &Ticket, deadline: Instant, heartbeat: &mut Heartbeat) -> Option<Outcome> {
    loop {
        heartbeat.beat();
        let slice = (Instant::now() + Duration::from_millis(1)).min(deadline);
        if let Some(outcome) = ticket.wait_until(slice) {
            return Some(outcome);
        }
        if Instant::now() >= deadline {
            return None;
        }
    }
}

/// The load's scratch fill words (spec 4.9): each round's convert lists
/// and the copy lists, which one layer's fill uses at a time.
pub(crate) struct ScratchFill {
    /// A flag per expert the launch routes to.
    pub flags: SharedWords,
    pub lists: Vec<[SharedWords; KINDS]>,
    pub copies: [SharedWords; KINDS],
    experts: usize,
    /// Records staged per round.
    round_records: usize,
}

impl ScratchFill {
    pub(crate) fn new(
        flags: SharedWords,
        lists: Vec<[SharedWords; KINDS]>,
        copies: [SharedWords; KINDS],
        experts: usize,
        round_records: usize,
    ) -> Self {
        flags.fill(0);
        for list in lists.iter().flatten().chain(&copies) {
            list.fill(0);
        }
        Self {
            flags,
            lists,
            copies,
            experts,
            round_records,
        }
    }

    fn rounds(&self) -> usize {
        self.lists.len()
    }

    fn acknowledgement(&self, round: usize) -> usize {
        SCRATCH_HEADER + round
    }

    fn arrival(&self, round: usize) -> usize {
        SCRATCH_HEADER + self.rounds() + round
    }
}

/// A fetched or fetching record: one staging record in each of the layer's
/// pools.
struct Staged {
    records: [u32; KINDS],
    ticket: Arc<Ticket>,
    /// The step sequence and layer whose resolve it was fetched for; a
    /// prefetch the resolve did not use is released once the device is past
    /// that layer.
    sequence: i32,
}

struct State {
    directory: SharedDirectory,
    /// Steps a shared-cache invalidation failed.
    failed_steps: std::collections::BTreeSet<i32>,
    layers: Vec<StreamedLayer>,
    scratch: ScratchFill,
    heartbeat: Heartbeat,
    pools: Vec<StagingPool>,
    /// (layer, expert) → its staged or fetching record.
    staged: HashMap<(usize, i32), Staged>,
    /// Records converted at (sequence, layer), with their expert: free once
    /// the device is past.
    converted: Vec<(i32, usize, [u32; KINDS], i32)>,
    /// (layer, expert) → free staging records still holding its stored
    /// bytes.
    kept: HashMap<(usize, i32), [u32; KINDS]>,
    /// The device's progress: the newest (sequence, layer) a hint proved.
    progress: (i32, usize),
}

/// The watcher's shared state and lifecycle.
pub(crate) struct StreamingRuntime {
    state: Arc<Mutex<State>>,
    source: Arc<ExpertSource>,
    in_flight: Arc<(AtomicUsize, Mutex<()>, Condvar)>,
    stopping: Arc<AtomicBool>,
    watcher: Option<JoinHandle<()>>,
}

impl StreamingRuntime {
    pub(crate) fn new(
        layers: Vec<StreamedLayer>,
        scratch: ScratchFill,
        heartbeat: Heartbeat,
        pools: Vec<StagingPool>,
        source: ExpertSource,
        directory: SharedDirectory,
    ) -> Result<Self, String> {
        let sizes = layers.iter().map(|l| (l.stamps.len(), l.commits.len())).collect::<Vec<_>>();
        directory.validate_ports(&sizes).map_err(|e| format!("shared directory ports: {e:?}"))?;
        let state = Arc::new(Mutex::new(State {
            directory,
            failed_steps: Default::default(),
            layers,
            scratch,
            heartbeat,
            pools,
            staged: HashMap::new(),
            converted: Vec::new(),
            kept: HashMap::new(),
            progress: (0, 0),
        }));
        let source = Arc::new(source);
        let in_flight = Arc::new((AtomicUsize::new(0), Mutex::new(()), Condvar::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let watcher = {
            let (state, source, in_flight, stopping) = (
                Arc::clone(&state),
                Arc::clone(&source),
                Arc::clone(&in_flight),
                Arc::clone(&stopping),
            );
            std::thread::Builder::new()
                .name("expert-watcher".into())
                .spawn(move || watch(&state, &source, &in_flight, &stopping))
                .map_err(|error| format!("expert watcher: {error}"))?
        };
        Ok(Self {
            state,
            source,
            in_flight,
            stopping,
            watcher: Some(watcher),
        })
    }

    /// A step is about to be submitted: the watcher polls until every
    /// submitted step has completed. Called before any of its launches is
    /// committed.
    pub(crate) fn step_submitted(&self) {
        let (count, lock, wake) = &*self.in_flight;
        let _guard = lock.lock().expect("watcher lock");
        count.fetch_add(1, Ordering::AcqRel);
        wake.notify_all();
    }

    /// A step completed. Its failure, if the watcher failed any of its
    /// requests, is returned and cleared (rule 6).
    pub(crate) fn step_completed(&self, sequence: i32, fenced: bool) -> Result<(), String> {
        self.in_flight.0.fetch_sub(1, Ordering::AcqRel);
        let mut state = self.state.lock().expect("streaming state");
        if !fenced {
            return Err("shared-cache step dropped without a completion fence".into());
        }
        let invalid = {
            let State { directory, layers, .. } = &mut *state;
            directory
                .retire_step(sequence, |layer, slot| layers[layer].commits.load(slot))
                .map_err(|e| format!("shared-cache completion: {e:?}"))?
        };
        invalidate(&mut state, invalid);
        let mut failed = state.failed_steps.remove(&sequence);
        for control in state
            .layers
            .iter()
            .flat_map(|layer| [&layer.control, &layer.scratch_control])
        {
            if control.load(FAILURE) == sequence {
                failed = true;
                control.store(FAILURE, 0);
            }
        }
        // Everything converted up to this step is in its slots now.
        if state.progress.0 < sequence {
            state.progress = (sequence, usize::MAX);
        }
        release_past(&mut state, &self.source);
        if failed {
            return Err(format!(
                "step {sequence}: an expert fetch failed or exceeded {PARK_BOUND:?}"
            ));
        }
        Ok(())
    }
}

impl Drop for StreamingRuntime {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.in_flight.2.notify_all();
        if let Some(watcher) = self.watcher.take() {
            let _ = watcher.join();
        }
    }
}

fn watch(
    state: &Mutex<State>,
    source: &ExpertSource,
    in_flight: &(AtomicUsize, Mutex<()>, Condvar),
    stopping: &AtomicBool,
) {
    let (count, lock, wake) = in_flight;
    loop {
        if stopping.load(Ordering::Acquire) {
            return;
        }
        if count.load(Ordering::Acquire) == 0 {
            let guard = lock.lock().expect("watcher lock");
            if count.load(Ordering::Acquire) == 0 && !stopping.load(Ordering::Acquire) {
                drop(wake.wait_timeout(guard, Duration::from_millis(50)));
            }
            continue;
        }
        let mut state = state.lock().expect("streaming state");
        state.heartbeat.beat();
        let progressed = poll(&mut state, source);
        drop(state);
        if !progressed {
            std::thread::sleep(IDLE);
        }
    }
}

/// One pass over every layer: new hints first (they also advance the clock),
/// then requests. True when it found anything.
fn poll(state: &mut State, source: &ExpertSource) -> bool {
    fence(Ordering::SeqCst);
    let mut progressed = false;
    let layers = state.layers.len();
    for layer in 0..layers {
        for distance in 0..2 {
            // Layer `l`'s ring `d` holds its hints for layer `l + d + 1`.
            let target = layer + distance + 1;
            if target >= layers {
                continue;
            }
            let sequence = state.layers[layer].rings[distance].load(0);
            if sequence == state.layers[layer].read[distance] {
                continue;
            }
            state.layers[layer].read[distance] = sequence;
            progressed = true;
            // A hint of layer `l` proves the device is past layer `l - 1`
            // of this step.
            advance(state, (sequence, layer), source);
            let ring = &state.layers[layer].rings[distance];
            let count = usize::try_from(ring.load(1)).unwrap_or(0).min(ring.len() - 2);
            let experts = (0..count).map(|i| ring.load(2 + i)).collect::<Vec<_>>();
            for expert in experts {
                prefetch(state, source, target, expert, sequence);
            }
        }
    }
    for layer in 0..layers {
        let sequence = state.layers[layer].control.load(REQUEST);
        if sequence != 0 && sequence != state.layers[layer].served {
            serve(state, source, layer, sequence);
            progressed = true;
        }
    }
    for layer in 0..layers {
        let sequence = state.layers[layer].scratch_control.load(REQUEST);
        if sequence != 0 && sequence != state.layers[layer].scratch_served {
            serve_scratch(state, source, layer, sequence);
            progressed = true;
        }
    }
    progressed
}

/// The device is at least at `point`: release what it no longer needs.
fn advance(state: &mut State, point: (i32, usize), source: &ExpertSource) {
    if point > state.progress {
        let invalid = {
            let State { directory, layers, .. } = &mut *state;
            directory
                .retire_progress(Clock { sequence: point.0, layer: point.1 }, |layer, slot| {
                    layers[layer].commits.load(slot)
                })
                .expect("device-published shared-cache progress")
        };
        invalidate(state, invalid);
        state.progress = point;
        release_past(state, source);
    }
}

/// Mappings the directory withdrew: unmap them and fail their steps.
fn invalidate(state: &mut State, invalid: Vec<(Clock, super::arena::Location)>) {
    for (clock, location) in invalid {
        state.failed_steps.insert(clock.sequence);
        let layer = &mut state.layers[location.key.layer.0 as usize];
        let expert = location.key.expert.0 as usize;
        if layer.slot_of[expert] == location.slot as i32 {
            layer.slot_of[expert] = -1;
            layer.slot_map.store(expert, -1);
            layer.expert_of[location.slot] = -1;
        }
        layer.control.store(FAILURE, clock.sequence);
    }
}

/// Free converted records and cancel or free unused prefetches of layers the
/// device has passed.
fn release_past(state: &mut State, source: &ExpertSource) {
    let progress = state.progress;
    let mut freed = Vec::new();
    state.converted.retain(|&(sequence, layer, records, expert)| {
        if (sequence, layer) < progress {
            freed.push((records, layer, expert));
            false
        } else {
            true
        }
    });
    for (records, layer, expert) in freed {
        free_records(state, layer, records, (expert >= 0).then_some(expert));
    }
    let stale = state
        .staged
        .iter()
        .filter(|((layer, _), staged)| (staged.sequence, *layer) < progress)
        .map(|(key, _)| *key)
        .collect::<Vec<_>>();
    for key in stale {
        let staged = &state.staged[&key];
        match staged.ticket.poll() {
            None => {
                // Still reading: cancel if it has not started, else let it end
                // and release it on a later pass.
                if source.cancel(&staged.ticket) {
                    let staged = state.staged.remove(&key).expect("present");
                    free_records(state, key.0, staged.records, None);
                }
            }
            Some(outcome) => {
                let staged = state.staged.remove(&key).expect("present");
                let holds = (outcome == Outcome::Done).then_some(key.1);
                free_records(state, key.0, staged.records, holds);
            }
        }
    }
}

/// Return a record's staging records to their pools. With `holds`, they
/// still hold that expert's stored bytes: a later prefetch or request takes
/// them back without a read (`revive`) until one of them is taken for
/// another record.
fn free_records(state: &mut State, layer: usize, records: [u32; KINDS], holds: Option<i32>) {
    if let Some(expert) = holds {
        forget(state, (layer, expert));
    }
    let pools = state.layers[layer].stored.map(|stored| stored.pool);
    for (kind, record) in records.into_iter().enumerate() {
        let pool = &mut state.pools[pools[kind]];
        pool.free.push_back(record);
        pool.holds[record as usize] = holds.map(|expert| (layer, expert));
    }
    if let Some(expert) = holds {
        state.kept.insert((layer, expert), records);
    }
}

/// Take `pool`'s longest-free record, forgetting what it held.
fn take_record(state: &mut State, pool: usize) -> Option<u32> {
    let record = state.pools[pool].free.pop_front()?;
    if let Some(key) = state.pools[pool].holds[record as usize].take() {
        forget(state, key);
    }
    Some(record)
}

/// A kept record is no longer intact.
fn forget(state: &mut State, key: (usize, i32)) {
    let Some(records) = state.kept.remove(&key) else {
        return;
    };
    let pools = state.layers[key.0].stored.map(|stored| stored.pool);
    for (kind, record) in records.into_iter().enumerate() {
        let holds = &mut state.pools[pools[kind]].holds[record as usize];
        if *holds == Some(key) {
            *holds = None;
        }
    }
}

/// Take back the staging records still holding `expert`'s stored bytes.
fn revive(state: &mut State, layer: usize, expert: i32) -> Option<[u32; KINDS]> {
    let records = state.kept.remove(&(layer, expert))?;
    let pools = state.layers[layer].stored.map(|stored| stored.pool);
    for (kind, record) in records.into_iter().enumerate() {
        let pool = &mut state.pools[pools[kind]];
        pool.holds[record as usize] = None;
        let index = pool
            .free
            .iter()
            .position(|&free| free == record)
            .expect("a kept record is free");
        pool.free.remove(index);
    }
    Some(records)
}

/// Reserve a staging record in each of the layer's pools and queue its
/// reads, or take back free records still holding the expert's bytes.
fn fetch(
    state: &mut State,
    source: &ExpertSource,
    layer: usize,
    expert: i32,
    priority: Priority,
    sequence: i32,
) -> bool {
    if let Some(records) = revive(state, layer, expert) {
        state.staged.insert(
            (layer, expert),
            Staged {
                records,
                ticket: Ticket::done(priority),
                sequence,
            },
        );
        return true;
    }
    let stored = state.layers[layer].stored;
    if stored
        .iter()
        .any(|tensor| state.pools[tensor.pool].free.is_empty())
    {
        return false;
    }
    let mut records = [0u32; KINDS];
    let mut ranges = Vec::with_capacity(KINDS);
    for (kind, tensor) in stored.iter().enumerate() {
        let record = take_record(state, tensor.pool).expect("checked free");
        let pool = &state.pools[tensor.pool];
        records[kind] = record;
        ranges.push(Range {
            offset: tensor.base + u64::try_from(expert).expect("an expert index") * tensor.bytes,
            bytes: usize::try_from(tensor.bytes).expect("a record fits the host"),
            destination: pool
                .bytes
                .at(u64::from(record) * pool.record_bytes, tensor.bytes),
        });
    }
    let ticket = source.submit(ranges, priority);
    state.staged.insert(
        (layer, expert),
        Staged {
            records,
            ticket,
            sequence,
        },
    );
    true
}

/// A hint: prefetch the expert unless it is resident, staged or fetching.
fn prefetch(state: &mut State, source: &ExpertSource, layer: usize, expert: i32, sequence: i32) {
    let Ok(index) = usize::try_from(expert) else {
        return;
    };
    if index >= state.layers[layer].slot_of.len()
        || state.layers[layer].slot_of[index] >= 0
        || state.staged.contains_key(&(layer, expert))
    {
        return;
    }
    // A prefetch never takes the records a demand read would need.
    if state.layers[layer]
        .stored
        .iter()
        .any(|tensor| state.pools[tensor.pool].free.len() <= state.pools[tensor.pool].reserve)
    {
        return;
    }
    fetch(state, source, layer, expert, Priority::Prefetch, sequence);
}

/// A parked step's request (spec 4.6 step 4): resolve it against the shared
/// directory, deliver every absent expert into staging, publish each choice's
/// placement and its acknowledgement as its record arrives, then the request's.
fn serve(state: &mut State, source: &ExpertSource, layer: usize, sequence: i32) {
    let started = Instant::now();
    let deadline = started + PARK_BOUND;
    advance(state, (sequence, layer), source);
    let width = state.layers[layer].placement.len() / 5;
    let count = usize::try_from(state.layers[layer].control.load(COUNT)).unwrap_or(0);
    let experts = state.layers[layer].slot_of.len();
    let fail = |state: &mut State| {
        state.layers[layer].control.store(FAILURE, sequence);
        state.layers[layer].served = sequence;
    };
    if count == 0 || count > width {
        fail(state);
        return;
    }
    let choices = (0..count)
        .map(|i| state.layers[layer].control.load(EXPERTS + i))
        .collect::<Vec<_>>();
    let routed = choices
        .iter()
        .map(|&word| {
            if word < 0 {
                -(i64::from(word)) - 1
            } else {
                i64::from(word)
            }
        })
        .collect::<Vec<_>>();
    if routed
        .iter()
        .any(|&expert| expert < 0 || expert as usize >= experts)
    {
        fail(state);
        return;
    }
    let routed = routed.into_iter().map(|e| e as i32).collect::<Vec<_>>();
    state
        .directory
        .sync_stamps(
            layer,
            (0..state.layers[layer].stamps.len()).map(|s| state.layers[layer].stamps.load(s)),
        )
        .expect("shared GPU stamps");
    let resolution = state
        .directory
        .begin(Clock { sequence, layer }, &routed)
        .expect("validated shared route union");
    for victim in &resolution.evicted {
        let owner = &mut state.layers[victim.key.layer.0 as usize];
        owner.slot_of[victim.key.expert.0 as usize] = -1;
        owner.slot_map.store(victim.key.expert.0 as usize, -1);
        owner.expert_of[victim.slot] = -1;
    }
    let mut pending = Vec::new();
    for choice in &resolution.choices {
        let expert = choice.location.key.expert.0 as i32;
        if !choice.needs_conversion || pending.contains(&expert) {
            continue;
        }
        pending.push(expert);
        let slot = choice.location.slot;
        state.layers[layer].commits.store(slot, 0);
        state.layers[layer].slot_of[expert as usize] = slot as i32;
        state.layers[layer].expert_of[slot] = expert;
        state.layers[layer].stamps.store(slot, sequence);
        match state.staged.get(&(layer, expert)) {
            Some(staged) => {
                if staged.ticket.priority() == Priority::Prefetch {
                    source.promote(&staged.ticket);
                }
            }
            None => {
                if !fetch(state, source, layer, expert, Priority::Demand, sequence) {
                    reclaim_prefetches(state, source, layer);
                    if !fetch(state, source, layer, expert, Priority::Demand, sequence) {
                        state
                            .directory
                            .submitted(Clock { sequence, layer })
                            .expect("failed immutable frame");
                        fail(state);
                        return;
                    }
                }
            }
        }
    }
    // All slots and raw source addresses become immutable before any source
    // readiness is published. Reserved source records remain staged until
    // their tickets finish, even when another record fails mid-request.
    for (c, choice) in resolution.choices.iter().enumerate() {
        if choices[c] < 0 {
            continue;
        }
        state.layers[layer]
            .placement
            .store(c, choice.location.slot as i32);
        let records = if choice.needs_conversion {
            state.staged[&(layer, routed[c])].records.map(|r| r as i32)
        } else {
            [-1; KINDS]
        };
        for (kind, record) in records.into_iter().enumerate() {
            state.layers[layer]
                .placement
                .store((1 + kind) * width + c, record);
        }
        state.layers[layer].placement.store(4 * width + c, 0);
    }
    state
        .directory
        .submitted(Clock { sequence, layer })
        .expect("immutable source publication");
    let mut outstanding = pending.clone();
    if outstanding.is_empty() {
        state.layers[layer].control.store(ACKNOWLEDGED, sequence);
    }
    for (c, choice) in resolution.choices.iter().enumerate() {
        if choices[c] >= 0 && !choice.needs_conversion {
            state.layers[layer].placement.store(4 * width + c, sequence);
        }
    }
    while !outstanding.is_empty() {
        state.heartbeat.beat();
        if Instant::now() >= deadline || state.layers[layer].control.load(FAILURE) == sequence {
            fail(state);
            return;
        }
        let mut arrived = false;
        let mut index = 0;
        while index < outstanding.len() {
            let expert = outstanding[index];
            match state.staged[&(layer, expert)].ticket.poll() {
                None => {
                    index += 1;
                    continue;
                }
                Some(Outcome::Done) => {}
                Some(_) => {
                    fail(state);
                    return;
                }
            }
            let staged = state
                .staged
                .remove(&(layer, expert))
                .expect("completed source record");
            state
                .converted
                .push((sequence, layer, staged.records, expert));
            outstanding.swap_remove(index);
            // The last source flag cannot let the gate pass finish before
            // the aggregate completion seal required by output publication.
            if outstanding.is_empty() {
                state.layers[layer].control.store(ACKNOWLEDGED, sequence);
            }
            for (c, choice) in resolution.choices.iter().enumerate() {
                if choices[c] >= 0 && choice.location.key.expert.0 == expert as u32 {
                    state.layers[layer].placement.store(4 * width + c, sequence);
                }
            }
            arrived = true;
        }
        if !arrived {
            std::thread::sleep(IDLE);
        }
    }
    state.layers[layer].served = sequence;
}

/// A scratch-served launch's fill (spec 4.9): list the routed experts in
/// the layer's bank for the copy, then stage the rest round by round, each
/// round's park bounded by `PARK_BOUND` from its start. Round `r`'s records
/// are freed once the device arrives at round `r + 1`, after converting
/// them; the last rounds' once the device is past the layer.
fn serve_scratch(state: &mut State, source: &ExpertSource, layer: usize, sequence: i32) {
    state.layers[layer].scratch_served = sequence;
    advance(state, (sequence, layer), source);
    let mut copies = Vec::new();
    let mut cold = Vec::new();
    for expert in 0..state.scratch.experts {
        if state.scratch.flags.load(expert) == 0 {
            continue;
        }
        match state.layers[layer].slot_of[expert] {
            slot if slot >= 0 => copies.push((slot, expert as i32)),
            _ => cold.push(expert as i32),
        }
    }
    let copies = copies
        .iter()
        .map(|&(slot, expert)| (slot as u32, expert as usize))
        .collect::<Vec<_>>();
    for list in &state.scratch.copies {
        publish_list(list, sequence, &copies);
    }
    let pools = state.layers[layer].stored.map(|stored| stored.pool);
    let chunks = cold
        .chunks(state.scratch.round_records.max(1))
        .map(<[i32]>::to_vec)
        .collect::<Vec<_>>();
    // Records of the rounds the device may still be converting, with the
    // expert each holds (-1 once a read of its round failed).
    let mut held: Vec<Vec<(i32, [u32; KINDS])>> = Vec::new();
    let mut failed = chunks.len() > state.scratch.rounds();
    for round in 0..state.scratch.rounds() {
        let deadline = Instant::now() + PARK_BOUND;
        if failed {
            break;
        }
        let experts = chunks.get(round).cloned().unwrap_or_default();
        // Round `round - 2`'s records are free once the device arrives at
        // round `round - 1`, past their convert.
        if round >= 2 && !experts.is_empty() {
            let arrival = state.scratch.arrival(round - 1);
            while state.layers[layer].scratch_control.load(arrival) != sequence {
                if Instant::now() >= deadline {
                    failed = true;
                    break;
                }
                state.heartbeat.beat();
                std::hint::spin_loop();
            }
            if failed {
                break;
            }
            for (expert, records) in std::mem::take(&mut held[round - 2]) {
                free_records(state, layer, records, (expert >= 0).then_some(expert));
            }
        }
        let mut records = Vec::with_capacity(experts.len());
        let mut ranges = Vec::with_capacity(experts.len() * KINDS);
        for &expert in &experts {
            if let Some(kept) = revive(state, layer, expert) {
                records.push((expert, kept));
                continue;
            }
            if pools.iter().any(|&pool| state.pools[pool].free.is_empty()) {
                reclaim_prefetches(state, source, usize::MAX);
            }
            if pools.iter().any(|&pool| state.pools[pool].free.is_empty()) {
                failed = true;
                break;
            }
            let mut record = [0u32; KINDS];
            let stored = state.layers[layer].stored;
            for (kind, tensor) in stored.iter().enumerate() {
                record[kind] = take_record(state, tensor.pool).expect("checked free");
                let pool = &state.pools[tensor.pool];
                ranges.push(Range {
                    offset: tensor.base
                        + u64::try_from(expert).expect("an expert index") * tensor.bytes,
                    bytes: usize::try_from(tensor.bytes).expect("a record fits the host"),
                    destination: pool
                        .bytes
                        .at(u64::from(record[kind]) * pool.record_bytes, tensor.bytes),
                });
            }
            records.push((expert, record));
        }
        if !ranges.is_empty() {
            let ticket = source.submit(ranges, Priority::Demand);
            failed |= wait_ticket(&ticket, deadline, &mut state.heartbeat) != Some(Outcome::Done);
        }
        if failed {
            held.push(records.into_iter().map(|(_, record)| (-1, record)).collect());
            break;
        }
        for (kind, list) in state.scratch.lists[round].iter().enumerate() {
            let entries = records
                .iter()
                .map(|&(expert, record)| (record[kind], expert as usize))
                .collect::<Vec<_>>();
            publish_list(list, sequence, &entries);
        }
        held.push(records);
        state.layers[layer]
            .scratch_control
            .store(state.scratch.acknowledgement(round), sequence);
    }
    if failed {
        state.layers[layer].scratch_control.store(FAILURE, sequence);
    }
    // The device may still read these until it is past the layer.
    for (expert, records) in held.into_iter().flatten() {
        state.converted.push((sequence, layer, records, expert));
    }
}

/// Write a convert or copy list for step `sequence`: its (source, target)
/// pairs, then the count, then the sequence, which the device checks against
/// its own step before it reads the rest (`routed_convert_wide`, `routed_copy`).
fn publish_list(list: &SharedWords, sequence: i32, entries: &[(u32, usize)]) {
    for (index, &(source, target)) in entries.iter().enumerate() {
        list.store(2 + 2 * index, source as i32);
        list.store(3 + 2 * index, target as i32);
    }
    list.store(1, entries.len() as i32);
    list.store(0, sequence);
}

/// Cancel every unstarted prefetch of the other layers so a demand read of
/// `layer` finds staging.
fn reclaim_prefetches(state: &mut State, source: &ExpertSource, layer: usize) {
    let keys = state
        .staged
        .iter()
        .filter(|(key, staged)| key.0 != layer && staged.ticket.priority() == Priority::Prefetch)
        .map(|(key, _)| *key)
        .collect::<Vec<_>>();
    for key in keys {
        // An arrived prefetch is a prediction's copy: given up like a queued
        // one. A started read runs to its end.
        let outcome = state.staged[&key].ticket.poll();
        if outcome.is_some() || source.cancel(&state.staged[&key].ticket) {
            let staged = state.staged.remove(&key).expect("present");
            let holds = (outcome == Some(Outcome::Done)).then_some(key.1);
            free_records(state, key.0, staged.records, holds);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog, Element, Tensor};
    fn words(device: &seismic::Device, n: u64) -> SharedWords {
        SharedWords::new(Tensor::zeros(device, Element::i32(), &[n]).unwrap()).unwrap()
    }
    #[test]
    fn first_record_publishes_before_last_source_and_neither_becomes_a_cache_hit() {
        publication_case(false);
        publication_case(true);
    }
    fn publication_case(fail_last: bool) {
        let device = DeviceCatalog::discover()
            .unwrap()
            .open_backend(BackendName::Metal)
            .unwrap();
        let control_tensor = Tensor::zeros(&device, Element::i32(), &[68]).unwrap();
        let placement_tensor = Tensor::zeros(&device, Element::i32(), &[5, 64]).unwrap();
        let control = SharedWords::new(control_tensor.clone()).unwrap();
        let placement = SharedWords::new(placement_tensor.clone()).unwrap();
        let layers = vec![StreamedLayer::new(
            words(&device, 256),
            words(&device, 8),
            words(&device, 8),
            SharedWords::new(control_tensor).unwrap(),
            SharedWords::new(placement_tensor).unwrap(),
            words(&device, 18),
            [words(&device, 66), words(&device, 66)],
            std::array::from_fn(|pool| StoredTensor {
                base: 0,
                bytes: 16,
                pool,
            }),
        )];
        let scratch = ScratchFill::new(
            words(&device, 256),
            vec![],
            [
                words(&device, 514),
                words(&device, 514),
                words(&device, 514),
            ],
            256,
            40,
        );
        let pools = (0..3)
            .map(|_| {
                StagingPool::new(
                    SharedBytes::new(Tensor::zeros(&device, Element::f32(), &[80, 2, 2]).unwrap())
                        .unwrap(),
                    80,
                    16,
                    16,
                )
            })
            .collect();
        let file =
            std::env::temp_dir().join(format!("expert-ready-control-{}", std::process::id()));
        std::fs::write(&file, [0u8; 512]).unwrap();
        let source =
            ExpertSource::new(crate::streaming::source::ModelFile::open(&file).unwrap()).unwrap();
        let runtime = StreamingRuntime::new(
            layers,
            scratch,
            Heartbeat::new(words(&device, 1)),
            pools,
            source,
            SharedDirectory::controlled(),
        )
        .unwrap();
        let last = Ticket::controlled();
        {
            let mut state = runtime.state.lock().unwrap();
            for expert in [0, 1] {
                let records = std::array::from_fn(|pool| take_record(&mut state, pool).unwrap());
                state.staged.insert(
                    (0, expert),
                    Staged {
                        records,
                        ticket: if expert == 0 {
                            Ticket::done(Priority::Demand)
                        } else {
                            Arc::clone(&last)
                        },
                        sequence: 1,
                    },
                );
            }
        }
        control.store(COUNT, 4);
        for (index, expert) in [0, 0, 1, 1].into_iter().enumerate() {
            control.store(EXPERTS + index, expert);
        }
        control.store(REQUEST, 1);
        runtime.step_submitted();
        let deadline = Instant::now() + Duration::from_secs(2);
        while placement.load(4 * 64) != 1 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(placement.load(4 * 64), 1);
        assert_eq!(placement.load(4 * 64 + 1), 1);
        assert_eq!(placement.load(4 * 64 + 2), 0);
        assert_eq!(control.load(ACKNOWLEDGED), 0);
        assert_eq!(control.load(FAILURE), 0);
        last.finish_controlled(fail_last);
        let terminal = if fail_last { FAILURE } else { ACKNOWLEDGED };
        while control.load(terminal) != 1 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(control.load(terminal), 1);
        // Wait for host publication to finish (the same mutex serializes
        // completion). Source completion has produced no conversion commits.
        {
            let state = runtime.state.lock().unwrap();
            for c in 0..4 {
                let expected = if fail_last && c >= 2 { 0 } else { 1 };
                assert_eq!(state.layers[0].placement.load(4 * 64 + c), expected);
            }
            assert_eq!(state.layers[0].slot_map.load(0), -1);
            assert_eq!(state.layers[0].slot_map.load(1), -1);
        }
        assert!(runtime.step_completed(1, true).is_err());
        drop(runtime);
        std::fs::remove_file(file).unwrap();
    }
}
