//! A device's recorded graphs of sealed-plan submissions (CUDA graphs,
//! Vulkan secondary command buffers, Metal schedules), by shape: keyed by
//! what fixes a graph's form (`ReplayKeys`), never by the storage it binds,
//! each shape keeping graphs by the parameters they bind (`bound`: on CUDA
//! and Vulkan every buffer address its launches take, on CUDA the run's
//! plan and encoded launches too). A run whose binding a graph of its shape
//! holds replays it as it is; another binding takes a graph of its shape
//! that may be updated now ([`RecordedGraph::updates_in_flight`]), updated
//! first; only where none may is a graph recorded, at a shape's first use
//! or once every graph of it is in flight. A CUDA graph is instantiated
//! over placeholder storage, bound to nothing until its first replay
//! ([`RecordedGraph::recorded_bound`]), and only once the device's queued
//! work has drained (`prepare`, [`Replays::record`]): every run, of any
//! place, slot, class or request, replays one, with no wait ever, since
//! CUDA updates only a graph whose launches completed, by its completion
//! event.
//!
//! The graphs kept are bounded by the device's graph budget
//! ([`super::GraphHold`]): what its owners hold, every graph their runs in
//! flight can need. A submission's own graphs past it break an invariant;
//! graphs beyond it that no submission needs are released, the least
//! recently replayed first, as a submission records and as a hold shrinks
//! ([`Replays::fit_to`]).

use crate::api::CallError;
use seismic_compiler::graph_cost::GraphCensus;
use std::any::Any;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

/// A recorded graph: what it is made of.
pub(crate) trait RecordedGraph {
    /// What it is made of ([`GraphCensus`]), which its memory follows.
    fn census(&self) -> GraphCensus;
    /// Whether it may be updated now to another binding: on CUDA once every
    /// launch of it queued so far completed (an update of one in flight
    /// waits for it), by its completion event; on Vulkan always, an update
    /// running in queue order beside its launches in flight
    /// ([`Self::updates_in_flight`]). Its fault where the device faulted.
    fn idle(&self) -> Result<bool, CallError>;
    /// Whether an update runs in queue order behind the graph's launches in
    /// flight, so one graph serves every run of its shape in a submission.
    fn updates_in_flight() -> bool;
    /// Whether a graph recorded for a run binds that run's parameters as
    /// recorded (a Vulkan division, a Metal schedule), rather than nothing
    /// until its first replay binds them (a CUDA graph, instantiated over
    /// placeholder storage).
    fn recorded_bound() -> bool {
        true
    }
}

pub(crate) struct Replays<G> {
    entries: Mutex<ReplayEntries<G>>,
    /// Graphs recorded so far: a submission that recorded one paid its
    /// recording on the host before its launches ran.
    recorded: std::sync::atomic::AtomicU64,
    /// Replays so far that bound other parameters than their graph's and
    /// updated it first.
    updated: std::sync::atomic::AtomicU64,
    /// The two counts by their graphs' keys' first word (a plan's, or on
    /// CUDA its layout's, `PlanShape::keyed`): what one load's plans cost,
    /// beside other loads on the device. Pruned when the last plan using a
    /// key drops ([`Self::unregister_plan`]).
    by_plan: Mutex<HashMap<u64, PlanCounts>>,
}

#[derive(Default)]
struct PlanCounts {
    live: usize,
    counts: GraphReplays,
}

impl<G> Default for Replays<G> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(ReplayEntries {
                clock: 0,
                next: 0,
                graphs: HashMap::new(),
                total_census: GraphCensus::default(),
                pending_drop: Vec::new(),
            }),
            recorded: std::sync::atomic::AtomicU64::new(0),
            updated: std::sync::atomic::AtomicU64::new(0),
            by_plan: Mutex::new(HashMap::new()),
        }
    }
}

/// One graph a submission replays: its shape's key, the launch parameters it
/// binds there (every buffer address its launches take, plus plan, encoded
/// mask and stamps on CUDA; none on a backend whose graphs hold no parameter),
/// and any timeline-word bindings (`word_bound`, empty on backends without
/// in-graph word nodes) that can be updated cheaply without rebinding launches.
pub(crate) struct Shaped {
    pub(crate) key: Vec<u64>,
    pub(crate) launch_bound: Vec<u64>,
    pub(crate) word_bound: Vec<u64>,
}

/// Which part of a recorded graph's binding changed and must be updated before
/// replay ([`Replays::replay`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplayUpdate {
    /// Launch parameters (and any timeline words) changed or were unbound.
    Launches,
    /// Only timeline-word bindings changed; launch parameters are unchanged.
    WordsOnly,
}

/// What a device's graphs cost serving so far: graphs recorded, and replays
/// that updated their graph's parameters first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphReplays {
    pub recorded: u64,
    pub updated: u64,
}

impl GraphReplays {
    /// The two counts together.
    pub fn plus(self, other: Self) -> Self {
        Self { recorded: self.recorded + other.recorded, updated: self.updated + other.updated }
    }

    /// What was counted since `earlier`, a reading of the same counts.
    pub fn since(self, earlier: Self) -> Self {
        Self {
            recorded: self.recorded.checked_sub(earlier.recorded).expect("graph counts only grow"),
            updated: self.updated.checked_sub(earlier.updated).expect("graph counts only grow"),
        }
    }
}

struct ReplayEntries<G> {
    clock: u64,
    /// The next graph's identity.
    next: u64,
    graphs: HashMap<Arc<[u64]>, Replay<G>>,
    /// Running census of every graph currently in `graphs`. Exact in `graphs`,
    /// `launches`, and `arguments`; `largest` is an upper bound (recomputed on
    /// the cold eviction path if an eviction ever makes it exceed `budget`).
    total_census: GraphCensus,
    /// Evicted graphs that were still in flight on a backend where
    /// `cuGraphExecDestroy` synchronizes while holding the driver context
    /// mutex (`!G::updates_in_flight()`). Reaped once `graph.idle() == Ok(true)`.
    pending_drop: Vec<Instance<G>>,
}

impl<G: RecordedGraph> ReplayEntries<G> {
    fn add_census(&mut self, census: GraphCensus) {
        self.total_census = self
            .total_census
            .plus(census)
            .expect("a device's recorded graphs' census fits u64");
    }

    fn sub_census(&mut self, census: GraphCensus) {
        self.total_census = self
            .total_census
            .minus(census)
            .expect("a kept graph is part of the census");
        if self.total_census.graphs == 0 {
            self.total_census = GraphCensus::default();
        }
    }

    fn replace_census(&mut self, before: GraphCensus, after: GraphCensus) {
        if before != after {
            self.sub_census(before);
            self.add_census(after);
        }
    }

    fn recompute_census(&mut self) -> GraphCensus {
        let exact = self
            .graphs
            .values()
            .flat_map(|replay| replay.graphs.iter())
            .try_fold(GraphCensus::default(), |census, instance| {
                census.plus(instance.graph.census())
            })
            .expect("a device's recorded graphs' census fits u64");
        self.total_census = exact;
        exact
    }
}

impl<G> ReplayEntries<G> {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    fn touch_first_mut(&mut self, key: &[u64]) -> Option<&mut G> {
        let replay = self.graphs.get_mut(key)?;
        self.clock += 1;
        let instance = replay.graphs.first_mut();
        instance.used = self.clock;
        Some(&mut instance.graph)
    }
}

/// Non-empty set of recorded graph instances for one shape.
struct ReplayGraphs<G> {
    first: Instance<G>,
    rest: Vec<Instance<G>>,
}

enum TakeOutcome<G> {
    /// Removed while leaving at least one graph in the shape.
    Taken(Instance<G>),
    /// The sole remaining graph in the shape matched; remove the shape entry.
    Last,
    /// No graph with that identity exists in this shape.
    Missing,
}

impl<G> ReplayGraphs<G> {
    fn singleton(first: Instance<G>) -> Self {
        Self { first, rest: Vec::new() }
    }

    fn len(&self) -> usize {
        1 + self.rest.len()
    }

    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    fn first_mut(&mut self) -> &mut Instance<G> {
        &mut self.first
    }

    fn push(&mut self, instance: Instance<G>) {
        self.rest.push(instance);
    }

    fn iter(&self) -> impl Iterator<Item = &Instance<G>> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Instance<G>> {
        std::iter::once(&mut self.first).chain(&mut self.rest)
    }

    fn find_mut(&mut self, identity: u64) -> Option<&mut Instance<G>> {
        self.iter_mut().find(|instance| instance.identity == identity)
    }

    fn take(&mut self, identity: u64) -> TakeOutcome<G> {
        if self.first.identity == identity {
            match self.rest.pop() {
                Some(replacement) => TakeOutcome::Taken(std::mem::replace(&mut self.first, replacement)),
                None => TakeOutcome::Last,
            }
        } else if let Some(at) = self.rest.iter().position(|instance| instance.identity == identity) {
            TakeOutcome::Taken(self.rest.swap_remove(at))
        } else {
            TakeOutcome::Missing
        }
    }
}

/// A shape's graphs.
struct Replay<G> {
    graphs: ReplayGraphs<G>,
    /// The plans the graphs' kernels belong to, kept loaded while they live;
    /// `None` for host-only schedules that hold no plan resources.
    _retained: Option<Arc<dyn Any + Send + Sync>>,
    /// The graphs of it kept for each launch in flight, one demand for each
    /// family that runs it, by its `flights` ([`Replays::ensure`]): a
    /// shape two families run is kept for both families' launches.
    pinned: Vec<Pinned>,
}

/// Graphs of a shape kept whatever the budget for one family: `per_flight`
/// for each of `flights`, its live slots, each a launch in flight. The
/// family's hold counts them (`LaunchGraphs::held`: its pinned graphs for
/// each launch beside one, and within the union for that one), so a
/// release never needs them.
struct Pinned {
    per_flight: usize,
    flights: Arc<std::sync::atomic::AtomicUsize>,
}

impl<G> Replay<G> {
    /// The graphs of it no release takes.
    fn pinned(&self) -> usize {
        self.pinned.iter().map(Pinned::kept).sum()
    }

    /// What every family but the one of `flights` keeps of it.
    fn pinned_beside(&self, flights: &Arc<std::sync::atomic::AtomicUsize>) -> usize {
        self.pinned.iter().filter(|pinned| !Arc::ptr_eq(&pinned.flights, flights)).map(Pinned::kept).sum()
    }
}

impl Pinned {
    fn kept(&self) -> usize {
        self.per_flight * self.flights.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// What parameters a recorded graph instance currently binds.
enum GraphBinding {
    /// Unknown or placeholder storage (not yet bound, or forgotten after a
    /// failed submission via [`Replays::unbind`]).
    Unbound,
    /// Bound to these launch and timeline-word parameters once the work
    /// queued so far ran.
    Bound {
        launch: Vec<u64>,
        word: Vec<u64>,
    },
}

impl GraphBinding {
    fn matches_exact(&self, launch_target: &[u64], word_target: &[u64]) -> bool {
        matches!(
            self,
            Self::Bound { launch, word }
                if launch.as_slice() == launch_target && word.as_slice() == word_target
        )
    }

    fn matches_launch(&self, launch_target: &[u64]) -> bool {
        matches!(self, Self::Bound { launch, .. } if launch.as_slice() == launch_target)
    }
}

struct Instance<G> {
    /// Its identity, which a submission's replays name it by
    /// ([`Replays::record`]).
    identity: u64,
    graph: G,
    /// The parameters it binds once the work queued so far ran; unbound where
    /// that is unknown (`Replays::unbind`).
    bound: GraphBinding,
    /// Clock of its last replay.
    used: u64,
}

/// Result of matching a submission's runs against kept graphs.
enum Chosen {
    /// Every run matched an existing graph.
    Complete(Vec<u64>),
    /// At least one run lacks a graph and must be recorded.
    Incomplete(Vec<Option<u64>>),
}

impl<G> Replays<G> {
    fn entries(&self) -> MutexGuard<'_, ReplayEntries<G>> {
        self.entries.lock().expect("native graph replays lock is never poisoned")
    }

    fn by_plan(&self) -> MutexGuard<'_, HashMap<u64, PlanCounts>> {
        self.by_plan.lock().expect("graph counts lock is never poisoned")
    }

    /// What the graphs cost serving so far.
    pub(crate) fn counts(&self) -> GraphReplays {
        GraphReplays {
            recorded: self.recorded.load(std::sync::atomic::Ordering::Relaxed),
            updated: self.updated.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// What the graphs of plan `plan`'s runs cost so far.
    pub(crate) fn plan_counts(&self, plan: u64) -> GraphReplays {
        self.by_plan()
            .get(&plan)
            .map_or(GraphReplays::default(), |entry| entry.counts)
    }

    /// Track a live plan key so its `by_plan` entry is pruned once every plan
    /// sharing that key has been dropped.
    pub(crate) fn register_plan(&self, plan: u64) {
        self.by_plan().entry(plan).or_default().live += 1;
    }

    /// Release a live plan key; removes its `by_plan` entry when no live plan
    /// remains for that key.
    pub(crate) fn unregister_plan(&self, plan: u64) {
        let mut by_plan = self.by_plan();
        if let Some(entry) = by_plan.get_mut(&plan) {
            entry.live = entry.live.saturating_sub(1);
            if entry.live == 0 {
                by_plan.remove(&plan);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn by_plan_len(&self) -> usize {
        self.by_plan().len()
    }

    /// Count a recording of a graph of `key` beside [`Self::record`]'s: one
    /// a backend makes again in replaying it (a Vulkan division formed
    /// again).
    pub(crate) fn count_recorded(&self, key: &[u64]) {
        self.recorded.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.of_plan(key, |counts| counts.recorded += 1);
    }

    fn count_updated(&self, key: &[u64]) {
        self.updated.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.of_plan(key, |counts| counts.updated += 1);
    }

    /// `count` on the counts of the plan `key` is a graph of.
    fn of_plan(&self, key: &[u64], count: impl FnOnce(&mut GraphReplays)) {
        let plan = *key.first().expect("a graph's key begins with its plan");
        count(&mut self.by_plan().entry(plan).or_default().counts);
    }
}

impl<G: RecordedGraph> Replays<G> {
    /// Choose, before a submission issues any work, the graph each of
    /// `graphs` replays, in order, and record those it lacks
    /// (`record(index)`, on Vulkan and Metal bound to that run's storage,
    /// on CUDA instantiated over placeholder storage, `cuda::record`), once
    /// `prepare` has run, outside the lock (CUDA's drain of the device's
    /// queued work), and the choice is made again. A run takes a graph of
    /// its shape bound to its storage; else one that may be updated now and
    /// that no other run of the submission takes, the one replayed longest
    /// ago; else a graph recorded for it. Graphs no run of the submission
    /// takes are released, the least recently replayed first, while the
    /// graphs exceed `budget`; the submission's own past it break an
    /// invariant. Each graph taken is the most recently used from then, so a
    /// hold that shrinks on another thread before the submission replays it
    /// ([`Self::fit_to`], a slot released off the submitting thread) releases
    /// every other graph first, and reaches it only where the submission's
    /// own graphs exceed what their holds bound, which breaks that invariant.
    /// Returns each run's graph's identity, which [`Self::replay`] takes.
    pub(crate) fn record(
        &self,
        graphs: &[Shaped],
        budget: GraphCensus,
        retained: &Arc<dyn Any + Send + Sync>,
        prepare: impl FnOnce() -> Result<(), CallError>,
        mut record: impl FnMut(usize) -> Result<G, CallError>,
    ) -> Result<Vec<u64>, CallError> {
        let mut entries = self.entries();
        let taken = match Self::choose(&mut entries, graphs)? {
            Chosen::Complete(taken) => taken,
            Chosen::Incomplete(_) => {
                drop(entries);
                prepare()?;
                entries = self.entries();
                match Self::choose(&mut entries, graphs)? {
                    Chosen::Complete(taken) => taken,
                    Chosen::Incomplete(partial) => {
                        // The rest record a graph each, which every later run
                        // of that binding takes, and where updates run in
                        // queue order every later run of its shape.
                        let mut resolved = Vec::with_capacity(graphs.len());
                        let mut by_key: HashMap<&[u64], u64> = HashMap::new();
                        let mut by_key_bound: HashMap<(&[u64], &[u64], &[u64]), u64> = HashMap::new();
                        for index in 0..graphs.len() {
                            let identity = match partial[index] {
                                Some(identity) => identity,
                                None => {
                                    let Shaped { key, launch_bound, word_bound } = &graphs[index];
                                    let existing = if G::updates_in_flight() {
                                        by_key.get(key.as_slice()).copied()
                                    } else {
                                        by_key_bound
                                            .get(&(key.as_slice(), launch_bound.as_slice(), word_bound.as_slice()))
                                            .copied()
                                    };
                                    match existing {
                                        Some(identity) => identity,
                                        None => {
                                            let graph = record(index)?;
                                            let binding = if G::recorded_bound() {
                                                GraphBinding::Bound {
                                                    launch: launch_bound.clone(),
                                                    word: word_bound.clone(),
                                                }
                                            } else {
                                                GraphBinding::Unbound
                                            };
                                            let identity = self.keep(&mut entries, key, graph, binding, retained);
                                            if G::updates_in_flight() {
                                                by_key.insert(key.as_slice(), identity);
                                            } else {
                                                by_key_bound.insert(
                                                    (key.as_slice(), launch_bound.as_slice(), word_bound.as_slice()),
                                                    identity,
                                                );
                                            }
                                            identity
                                        }
                                    }
                                }
                            };
                            resolved.push(identity);
                        }
                        resolved
                    }
                }
            }
        };
        Self::fit(&mut entries, &taken, budget);
        Ok(taken)
    }

    /// Like [`Self::record`], except when every unmatched run's shape already
    /// has at least one untaken recorded graph currently in flight, skip
    /// `prepare` and recording and return `None` for those busy runs so the
    /// caller can launch them directly without draining the device. Runs whose
    /// shape has no graph yet (or needs more graphs than exist for distinct
    /// runs within this submission) still call `prepare` and record normally.
    pub(crate) fn record_or_direct(
        &self,
        graphs: &[Shaped],
        budget: GraphCensus,
        retained: &Arc<dyn Any + Send + Sync>,
        prepare: impl FnOnce() -> Result<(), CallError>,
        mut record: impl FnMut(usize) -> Result<G, CallError>,
    ) -> Result<Vec<Option<u64>>, CallError> {
        let mut entries = self.entries();
        let taken: Vec<Option<u64>> = match Self::choose(&mut entries, graphs)? {
            Chosen::Complete(taken) => taken.into_iter().map(Some).collect(),
            Chosen::Incomplete(partial) => {
                if !Self::needs_recording(&entries, graphs, &partial) {
                    partial
                } else {
                    drop(entries);
                    prepare()?;
                    entries = self.entries();
                    match Self::choose(&mut entries, graphs)? {
                        Chosen::Complete(taken) => taken.into_iter().map(Some).collect(),
                        Chosen::Incomplete(partial) => {
                            let mut resolved: Vec<Option<u64>> = Vec::with_capacity(graphs.len());
                            let mut capacity_exhausted = false;
                            let mut by_key: HashMap<&[u64], u64> = HashMap::new();
                            let mut by_key_bound: HashMap<(&[u64], &[u64], &[u64]), u64> = HashMap::new();
                            for index in 0..graphs.len() {
                                let identity = match partial[index] {
                                    Some(identity) => Some(identity),
                                    None if capacity_exhausted => None,
                                    None => {
                                        let Shaped { key, launch_bound, word_bound } = &graphs[index];
                                        let existing = if G::updates_in_flight() {
                                            by_key.get(key.as_slice()).copied()
                                        } else {
                                            by_key_bound
                                                .get(&(key.as_slice(), launch_bound.as_slice(), word_bound.as_slice()))
                                                .copied()
                                        };
                                        if let Some(identity) = existing {
                                            Some(identity)
                                        } else {
                                            let recorded = match record(index) {
                                                Ok(graph) => Some(graph),
                                                Err(ref error) if is_capacity_refusal(error) => {
                                                    // Evict untaken, unpinned cached graphs incrementally
                                                    // (oldest half first so hot decode/draft graphs survive,
                                                    // falling back to full eviction only if a retry still
                                                    // refuses) before falling back to direct launch.
                                                    let keep: Vec<u64> = resolved
                                                        .iter()
                                                        .chain(partial.iter().skip(index))
                                                        .flatten()
                                                        .copied()
                                                        .collect();
                                                    let half_graphs =
                                                        (entries.total_census.graphs / 2).max(keep.len() as u64);
                                                    let targets = [
                                                        GraphCensus {
                                                            graphs: half_graphs,
                                                            launches: u64::MAX,
                                                            arguments: u64::MAX,
                                                            largest: u64::MAX,
                                                        },
                                                        GraphCensus::default(),
                                                    ];
                                                    let mut recovered = None;
                                                    for target in targets {
                                                        let (released, _) = Self::release_beyond(
                                                            &mut entries,
                                                            &keep,
                                                            GraphCensus::default(),
                                                            target,
                                                        );
                                                        if released.is_empty() {
                                                            continue;
                                                        }
                                                        Self::defer_or_drop(&mut entries, released);
                                                        match record(index) {
                                                            Ok(graph) => {
                                                                recovered = Some(graph);
                                                                break;
                                                            }
                                                            Err(ref err) if is_capacity_refusal(err) => continue,
                                                            Err(err) => return Err(err),
                                                        }
                                                    }
                                                    if recovered.is_none() {
                                                        capacity_exhausted = true;
                                                    }
                                                    recovered
                                                }
                                                Err(error) => return Err(error),
                                            };
                                            match recorded {
                                                Some(graph) => {
                                                    let binding = if G::recorded_bound() {
                                                        GraphBinding::Bound {
                                                            launch: launch_bound.clone(),
                                                            word: word_bound.clone(),
                                                        }
                                                    } else {
                                                        GraphBinding::Unbound
                                                    };
                                                    let identity = self.keep(&mut entries, key, graph, binding, retained);
                                                    if G::updates_in_flight() {
                                                        by_key.insert(key.as_slice(), identity);
                                                    } else {
                                                        by_key_bound.insert(
                                                            (key.as_slice(), launch_bound.as_slice(), word_bound.as_slice()),
                                                            identity,
                                                        );
                                                    }
                                                    Some(identity)
                                                }
                                                None => None,
                                            }
                                        }
                                    }
                                };
                                resolved.push(identity);
                            }
                            resolved
                        }
                    }
                }
            }
        };
        let ids: Vec<u64> = taken.iter().flatten().copied().collect();
        Self::fit(&mut entries, &ids, budget);
        Ok(taken)
    }

    /// Whether any unmatched run in `partial` genuinely requires recording a
    /// new graph (either its shape has no graph yet, or every existing graph
    /// of its shape is already taken by another run of this submission) rather
    /// than merely waiting on an untaken in-flight graph.
    fn needs_recording(entries: &ReplayEntries<G>, graphs: &[Shaped], partial: &[Option<u64>]) -> bool {
        let taken_ids: HashSet<u64> = partial.iter().flatten().copied().collect();
        for (index, slot) in partial.iter().enumerate() {
            if slot.is_some() {
                continue;
            }
            let Some(replay) = entries.graphs.get(graphs[index].key.as_slice()) else {
                return true;
            };
            let has_untaken_in_flight = replay
                .graphs
                .iter()
                .any(|instance| !taken_ids.contains(&instance.identity));
            if !has_untaken_in_flight {
                return true;
            }
        }
        false
    }

    /// Single-lock lookup and use for an unbound, host-only graph (a Metal
    /// schedule): zero allocations on cache hit, does not retain plan objects
    /// or pollute `by_plan`, and derives outside the lock on a miss.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn with_unbound(
        &self,
        key: &[u64],
        budget: GraphCensus,
        derive: impl FnOnce() -> G,
        use_graph: impl FnOnce(&mut G) -> Result<(), CallError>,
    ) -> Result<(), CallError> {
        let mut entries = self.entries();
        if let Some(graph) = entries.touch_first_mut(key) {
            let before = graph.census();
            let result = use_graph(graph);
            let after = graph.census();
            entries.replace_census(before, after);
            return result;
        }
        drop(entries);
        let graph = derive();
        let mut entries = self.entries();
        if let Some(graph) = entries.touch_first_mut(key) {
            let before = graph.census();
            let result = use_graph(graph);
            let after = graph.census();
            entries.replace_census(before, after);
            return result;
        }
        let incoming = graph.census();
        Self::fit_with(&mut entries, &[], incoming, budget);
        let identity = entries.next;
        entries.next += 1;
        entries.clock += 1;
        let used = entries.clock;
        entries.add_census(incoming);
        let replay = entries.graphs.entry(Arc::from(key)).or_insert(Replay {
            graphs: ReplayGraphs::singleton(Instance {
                identity,
                graph,
                bound: GraphBinding::Bound {
                    launch: Vec::new(),
                    word: Vec::new(),
                },
                used,
            }),
            _retained: None,
            pinned: Vec::new(),
        });
        let graph = &mut replay.graphs.first_mut().graph;
        let before = graph.census();
        let result = use_graph(graph);
        let after = graph.census();
        entries.replace_census(before, after);
        result
    }

    /// Keep `per_flight` graphs of shape `key` for each of `flights`, those
    /// a launch of `per_flight` runs of it takes for each launch in flight,
    /// beside what every other family running it keeps, recorded ahead of
    /// them (`record`, after `prepare`, which runs only where one is to be
    /// recorded), each bound to nothing until its first replay; no release
    /// takes them while `flights` stands ([`Pinned`]), and this demand
    /// replaces only the one of the same `flights`, its family's.
    /// Graphs beyond `budget` that it does not keep are released, the least
    /// recently replayed first; its own past it break an invariant.
    pub(crate) fn ensure(
        &self,
        key: &[u64],
        per_flight: usize,
        flights: &Arc<std::sync::atomic::AtomicUsize>,
        budget: GraphCensus,
        retained: &Arc<dyn Any + Send + Sync>,
        prepare: impl FnOnce() -> Result<(), seismic_compiler::errors::ExecutionError>,
        mut record: impl FnMut() -> Result<G, seismic_compiler::errors::ExecutionError>,
    ) -> Result<(), seismic_compiler::errors::ExecutionError> {
        let kept = |entries: &ReplayEntries<G>| entries.graphs.get(key).map_or(0, |replay| replay.graphs.len());
        let beside = |entries: &ReplayEntries<G>| entries.graphs.get(key).map_or(0, |replay| replay.pinned_beside(flights));
        let mut entries = self.entries();
        let mut count = beside(&entries) + per_flight * flights.load(std::sync::atomic::Ordering::Acquire);
        if kept(&entries) < count {
            drop(entries);
            prepare()?;
            entries = self.entries();
            count = beside(&entries) + per_flight * flights.load(std::sync::atomic::Ordering::Acquire);
        }
        for _ in kept(&entries)..count {
            let graph = record()?;
            self.keep(&mut entries, key, graph, GraphBinding::Unbound, retained);
        }
        if let Some(replay) = entries.graphs.get_mut(key) {
            replay.pinned.retain(|pinned| !Arc::ptr_eq(&pinned.flights, flights));
            replay.pinned.push(Pinned { per_flight, flights: flights.clone() });
        }
        let own = entries
            .graphs
            .get(key)
            .map(|replay| replay.graphs.iter().map(|instance| instance.identity).collect::<Vec<_>>())
            .unwrap_or_default();
        Self::fit(&mut entries, &own, budget);
        Ok(())
    }

    /// Keep `graph`, just recorded, of shape `key`, binding `bound` (unbound
    /// until its first replay binds it): counted recorded, and the most
    /// recently used, so it is released after every graph kept before it.
    /// Returns its identity.
    fn keep(
        &self,
        entries: &mut ReplayEntries<G>,
        key: &[u64],
        graph: G,
        bound: GraphBinding,
        retained: &Arc<dyn Any + Send + Sync>,
    ) -> u64 {
        self.count_recorded(key);
        let identity = entries.next;
        entries.next += 1;
        entries.clock += 1;
        let used = entries.clock;
        entries.add_census(graph.census());
        let instance = Instance { identity, graph, bound, used };
        if let Some(replay) = entries.graphs.get_mut(key) {
            replay.graphs.push(instance);
        } else {
            entries.graphs.insert(
                Arc::from(key),
                Replay {
                    graphs: ReplayGraphs::singleton(instance),
                    _retained: Some(retained.clone()),
                    pinned: Vec::new(),
                },
            );
        }
        identity
    }

    /// The graph each of `graphs` takes of those kept ([`Self::record`]),
    /// stamping each chosen graph as most recently used; returns
    /// [`Chosen::Incomplete`] where any run lacks a graph and must record one.
    ///
    /// Matches in three tiers, each O(R) over `graphs`:
    /// 1. Exact match on `(launch_bound, word_bound)` -> zero updates.
    /// 2. Match on `launch_bound` alone among idle, untaken graphs -> only
    ///    timeline-word nodes are updated (`ReplayUpdate::WordsOnly`), skipping
    ///    kernel parameter rebinding.
    /// 3. LRU fallback among remaining idle, untaken graphs.
    fn choose(entries: &mut ReplayEntries<G>, graphs: &[Shaped]) -> Result<Chosen, CallError> {
        entries.clock += 1;
        let clock = entries.clock;
        let mut taken: Vec<Option<u64>> = vec![None; graphs.len()];
        let mut taken_ids: HashSet<u64> = HashSet::new();
        let mut unmatched = graphs.len();
        // Tier 1: exact match on `(launch_bound, word_bound)`. Every run of
        // that exact binding shares the graph (requiring `idle()` on backends
        // where launching or updating an in-flight graph can block while
        // holding the driver context mutex).
        for (index, shaped) in graphs.iter().enumerate() {
            if let Some(replay) = entries.graphs.get_mut(shaped.key.as_slice()) {
                for instance in replay.graphs.iter_mut() {
                    if instance.bound.matches_exact(&shaped.launch_bound, &shaped.word_bound)
                        && (G::updates_in_flight() || instance.graph.idle()?)
                    {
                        instance.used = clock;
                        taken[index] = Some(instance.identity);
                        taken_ids.insert(instance.identity);
                        unmatched -= 1;
                        break;
                    }
                }
            }
        }
        if unmatched == 0 {
            return Ok(Chosen::Complete(
                taken.into_iter().map(|id| id.expect("all runs matched")).collect(),
            ));
        }
        // Tiers 2 & 3: another takes a graph of its shape that may be updated
        // now — one no other run takes, unless updates run in queue order.
        // Unmatched runs sharing a key (or `(key, launch_bound, word_bound)`
        // when updates cannot overlap flight) share the same chosen graph in
        // O(R) per tier.
        if G::updates_in_flight() {
            let mut assigned_by_key: HashMap<&[u64], Option<u64>> = HashMap::new();
            for index in 0..graphs.len() {
                if taken[index].is_some() {
                    continue;
                }
                let key = graphs[index].key.as_slice();
                let chosen = match assigned_by_key.get(key) {
                    Some(&cached) => cached,
                    None => {
                        let picked = entries.graphs.get_mut(key).and_then(|replay| {
                            let best = replay.graphs.iter_mut().min_by_key(|instance| instance.used)?;
                            best.used = clock;
                            Some(best.identity)
                        });
                        assigned_by_key.insert(key, picked);
                        picked
                    }
                };
                if let Some(identity) = chosen {
                    taken[index] = Some(identity);
                    unmatched -= 1;
                }
            }
        } else {
            // Tier 2: match on `launch_bound` alone among idle, untaken graphs.
            // Checked before `idle()` so non-matching graphs incur zero driver
            // event queries, and run across all unmatched runs before Tier 3 so
            // LRU fallback never steals a graph that another run matches on
            // `launch_bound`.
            let mut tier2_assigned: HashMap<(&[u64], &[u64], &[u64]), Option<u64>> = HashMap::new();
            for index in 0..graphs.len() {
                if taken[index].is_some() {
                    continue;
                }
                let key = graphs[index].key.as_slice();
                let launch = graphs[index].launch_bound.as_slice();
                let word = graphs[index].word_bound.as_slice();
                let chosen = match tier2_assigned.get(&(key, launch, word)) {
                    Some(&cached) => cached,
                    None => {
                        let mut picked = None;
                        if let Some(replay) = entries.graphs.get_mut(key) {
                            let mut free: Option<&mut Instance<G>> = None;
                            for instance in replay.graphs.iter_mut() {
                                if !taken_ids.contains(&instance.identity)
                                    && instance.bound.matches_launch(launch)
                                    && instance.graph.idle()?
                                    && free.as_ref().is_none_or(|best| instance.used < best.used)
                                {
                                    free = Some(instance);
                                }
                            }
                            if let Some(instance) = free {
                                instance.used = clock;
                                taken_ids.insert(instance.identity);
                                picked = Some(instance.identity);
                            }
                        }
                        tier2_assigned.insert((key, launch, word), picked);
                        picked
                    }
                };
                if let Some(identity) = chosen {
                    taken[index] = Some(identity);
                    unmatched -= 1;
                }
            }
            if unmatched == 0 {
                return Ok(Chosen::Complete(
                    taken.into_iter().map(|id| id.expect("all runs matched")).collect(),
                ));
            }
            // Tier 3: LRU fallback among remaining idle, untaken graphs.
            let mut tier3_assigned: HashMap<(&[u64], &[u64], &[u64]), Option<u64>> = HashMap::new();
            for index in 0..graphs.len() {
                if taken[index].is_some() {
                    continue;
                }
                let key = graphs[index].key.as_slice();
                let launch = graphs[index].launch_bound.as_slice();
                let word = graphs[index].word_bound.as_slice();
                let chosen = match tier3_assigned.get(&(key, launch, word)) {
                    Some(&cached) => cached,
                    None => {
                        let mut picked = None;
                        if let Some(replay) = entries.graphs.get_mut(key) {
                            let mut free: Option<&mut Instance<G>> = None;
                            for instance in replay.graphs.iter_mut() {
                                if !taken_ids.contains(&instance.identity)
                                    && instance.graph.idle()?
                                    && free.as_ref().is_none_or(|best| instance.used < best.used)
                                {
                                    free = Some(instance);
                                }
                            }
                            if let Some(instance) = free {
                                instance.used = clock;
                                taken_ids.insert(instance.identity);
                                picked = Some(instance.identity);
                            }
                        }
                        tier3_assigned.insert((key, launch, word), picked);
                        picked
                    }
                };
                if let Some(identity) = chosen {
                    taken[index] = Some(identity);
                    unmatched -= 1;
                }
            }
        }
        if unmatched == 0 {
            Ok(Chosen::Complete(
                taken.into_iter().map(|id| id.expect("all runs matched")).collect(),
            ))
        } else {
            Ok(Chosen::Incomplete(taken))
        }
    }

    /// Release graphs while they exceed `budget`, the least recently
    /// replayed first: as a hold of it shrinks, so the graphs never hold
    /// more than their owners do.
    pub(crate) fn fit_to(&self, budget: GraphCensus) {
        let released = {
            let mut entries = self.entries();
            Self::release_beyond(&mut entries, &[], GraphCensus::default(), budget).0
        };
        // Dropped after the lock, as what they keep may drop holds.
        drop(released);
    }

    /// Run `replay` with graph `identity` of `shaped`'s shape, which
    /// [`Self::record`] chose for this run, having first run `update` on it
    /// where it binds other parameters than `shaped`'s: `ReplayUpdate::Launches`
    /// when launch parameters changed or were unbound (counted in `updated`),
    /// `ReplayUpdate::WordsOnly` when only timeline-word bindings changed (not
    /// counted in `updated`, as launches are not rebound).
    pub(crate) fn replay(
        &self,
        shaped: &Shaped,
        identity: u64,
        update: impl FnOnce(&mut G, ReplayUpdate) -> Result<(), CallError>,
        replay: impl FnOnce(&mut G) -> Result<(), CallError>,
    ) -> Result<(), CallError> {
        let mut entries = self.entries();
        let entries = &mut *entries;
        entries.clock += 1;
        let clock = entries.clock;
        let instance = entries
            .graphs
            .get_mut(shaped.key.as_slice())
            .and_then(|replay| replay.graphs.find_mut(identity))
            .expect("a submission replays only graphs chosen for it");
        instance.used = clock;
        let before = instance.graph.census();
        let outcome = (|| {
            if !instance.bound.matches_exact(&shaped.launch_bound, &shaped.word_bound) {
                let kind = if instance.bound.matches_launch(&shaped.launch_bound) {
                    ReplayUpdate::WordsOnly
                } else {
                    ReplayUpdate::Launches
                };
                // Unknown until the update is queued: a failure in it leaves the
                // graph to be updated whole.
                instance.bound = GraphBinding::Unbound;
                update(&mut instance.graph, kind)?;
                instance.bound = GraphBinding::Bound {
                    launch: shaped.launch_bound.clone(),
                    word: shaped.word_bound.clone(),
                };
                if kind == ReplayUpdate::Launches {
                    self.count_updated(&shaped.key);
                }
            }
            replay(&mut instance.graph)
        })();
        let after = instance.graph.census();
        entries.replace_census(before, after);
        outcome
    }

    /// Drop every graph if `unheld` says, under this lock, that no hold of
    /// the device's graph budget is live (`GraphHold::drop`): none can then
    /// be replayed, and each keeps its plans, which hold the device. A hold
    /// taken after the check records only once this lock is released. The
    /// graphs drop after it is, as what they keep may drop holds.
    pub(crate) fn clear_unheld(&self, unheld: impl FnOnce() -> bool) {
        let graphs = {
            let mut entries = self.entries();
            if !unheld() {
                return;
            }
            entries.total_census = GraphCensus::default();
            std::mem::take(&mut entries.graphs)
        };
        drop(graphs);
    }

    /// Forget what the graphs of `graphs`' shapes bind, after their
    /// submission failed before every update it queued was committed: an
    /// update carried in queued work (Vulkan's) runs only with that work, so
    /// a graph's next replay updates it whole, `forget` dropping the graph's
    /// own record of its bindings.
    pub(crate) fn unbind(&self, graphs: &[Shaped], mut forget: impl FnMut(&mut G)) {
        let mut entries = self.entries();
        let mut changed = false;
        for shaped in graphs {
            if let Some(replay) = entries.graphs.get_mut(shaped.key.as_slice()) {
                for instance in replay.graphs.iter_mut() {
                    instance.bound = GraphBinding::Unbound;
                    let before = instance.graph.census();
                    forget(&mut instance.graph);
                    if instance.graph.census() != before {
                        changed = true;
                    }
                }
            }
        }
        if changed {
            entries.recompute_census();
        }
    }

    /// Fit the graphs to `budget` now that the submission taking `taken` has
    /// replayed: a graph's census can change as it is replayed (a Vulkan
    /// division recorded again, `vulkan::Divided`).
    pub(crate) fn fit_after_replay(&self, taken: &[u64], budget: GraphCensus) {
        let mut entries = self.entries();
        Self::fit(&mut entries, taken, budget);
    }

    /// Release graphs no run of the submission takes (`taken`), the least
    /// recently replayed first, while the graphs exceed `budget`. The budget
    /// holds every graph its owners' runs in flight can need, so the submission's
    /// own past it break an invariant.
    fn fit(entries: &mut ReplayEntries<G>, taken: &[u64], budget: GraphCensus) {
        Self::fit_with(entries, taken, GraphCensus::default(), budget);
    }

    fn drain_idle_drops(entries: &mut ReplayEntries<G>) {
        if entries.pending_drop.is_empty() {
            return;
        }
        let mut ready = Vec::new();
        let mut idx = 0;
        while idx < entries.pending_drop.len() {
            if G::updates_in_flight()
                || entries.pending_drop[idx].graph.idle().unwrap_or(false)
            {
                ready.push(entries.pending_drop.swap_remove(idx));
            } else {
                idx += 1;
            }
        }
        drop(ready);
    }

    fn defer_or_drop(entries: &mut ReplayEntries<G>, released: Vec<Instance<G>>) {
        if G::updates_in_flight() {
            drop(released);
            return;
        }
        let mut ready = Vec::new();
        for instance in released {
            if instance.graph.idle().unwrap_or(false) {
                ready.push(instance);
            } else {
                entries.pending_drop.push(instance);
            }
        }
        drop(ready);
    }

    fn fit_with(entries: &mut ReplayEntries<G>, taken: &[u64], incoming: GraphCensus, budget: GraphCensus) {
        Self::drain_idle_drops(entries);
        let (released, held) = Self::release_beyond(entries, taken, incoming, budget);
        assert!(
            held.within(budget),
            "a submission's graphs, {held:?}, exceed every graph its owners hold, {budget:?}"
        );
        Self::defer_or_drop(entries, released);
    }

    /// Take out graphs no run of a submission takes (`taken`), the least
    /// recently replayed first, while the kept graphs plus `incoming` exceed
    /// `budget`; returns them, to be dropped, and the census of those kept
    /// (including `incoming`).
    fn release_beyond(
        entries: &mut ReplayEntries<G>,
        taken: &[u64],
        incoming: GraphCensus,
        budget: GraphCensus,
    ) -> (Vec<Instance<G>>, GraphCensus) {
        let fast_held = entries
            .total_census
            .plus(incoming)
            .expect("a device's recorded graphs' census fits u64");
        if fast_held.within(budget) {
            return (Vec::new(), fast_held);
        }
        // Recompute exact census on the cold path so a stale `largest` bound
        // from an earlier eviction cannot trigger unnecessary evictions.
        let mut held = entries
            .recompute_census()
            .plus(incoming)
            .expect("a device's recorded graphs' census fits u64");
        let mut released = Vec::new();
        if held.within(budget) {
            return (released, held);
        }
        let taken_set: HashSet<u64> = taken.iter().copied().collect();
        let mut others = entries
            .graphs
            .iter()
            .flat_map(|(key, replay)| replay.graphs.iter().map(move |instance| (instance.used, instance.identity, key.clone())))
            .filter(|(_, identity, _)| !taken_set.contains(identity))
            .collect::<Vec<_>>();
        others.sort_unstable();
        for (_, identity, key) in others {
            let Entry::Occupied(mut entry) = entries.graphs.entry(key) else {
                unreachable!("an other graph's shape is recorded");
            };
            // A shape's graphs kept for its launches in flight stay.
            if entry.get().graphs.len() <= entry.get().pinned() {
                continue;
            }
            let instance = match entry.get_mut().graphs.take(identity) {
                TakeOutcome::Taken(instance) => instance,
                TakeOutcome::Last => entry.remove().graphs.first,
                TakeOutcome::Missing => unreachable!("an other graph is kept"),
            };
            let census = instance.graph.census();
            entries.sub_census(census);
            held = held.minus(census).expect("a kept graph is part of the census");
            released.push(instance);
            if held.within(budget) {
                break;
            }
        }
        (released, held)
    }
}

/// Whether `error` is a memory-capacity refusal from the driver or ledger
/// during graph recording, which [`Replays::record_or_direct`] can recover
/// from by evicting untaken cached graphs and/or falling back to direct
/// stream-ordered launch.
fn is_capacity_refusal(error: &CallError) -> bool {
    matches!(
        error,
        CallError::Execution(
            seismic_compiler::errors::ExecutionError::DeviceRefused { .. }
                | seismic_compiler::errors::ExecutionError::AllocationCapacity { .. }
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph of a census, the parameters it was last bound to, and whether
    /// its launches completed; `QUEUED` whether its updates run in queue
    /// order (Vulkan) rather than only when it is idle (CUDA).
    struct Graph<const QUEUED: bool>(GraphCensus, Vec<u64>, Arc<std::sync::atomic::AtomicBool>);

    thread_local! {
        static CENSUS_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static IDLE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    impl<const QUEUED: bool> RecordedGraph for Graph<QUEUED> {
        fn census(&self) -> GraphCensus {
            CENSUS_CALLS.with(|c| c.set(c.get() + 1));
            self.0
        }
        fn idle(&self) -> Result<bool, CallError> {
            IDLE_CALLS.with(|c| c.set(c.get() + 1));
            Ok(self.2.load(std::sync::atomic::Ordering::Relaxed))
        }
        fn updates_in_flight() -> bool {
            QUEUED
        }
    }

    fn graph<const QUEUED: bool>(launches: u64) -> Graph<QUEUED> {
        Graph(
            GraphCensus { graphs: 1, launches, arguments: 16 * launches, largest: 16 },
            Vec::new(),
            Arc::new(std::sync::atomic::AtomicBool::new(true)),
        )
    }

    fn retained() -> Arc<dyn Any + Send + Sync> {
        Arc::new(())
    }

    fn shaped(key: u64, bound: u64) -> Shaped {
        Shaped {
            key: vec![key],
            launch_bound: vec![bound],
            word_bound: Vec::new(),
        }
    }

    fn shaped_with_word(key: u64, launch: u64, word_addr: u64, word_val: u64) -> Shaped {
        Shaped {
            key: vec![key],
            launch_bound: vec![launch],
            word_bound: vec![word_addr, word_val],
        }
    }

    /// Record and replay `runs` as one submission, each replay updating its
    /// graph to its run's binding; returns what each replay was bound to.
    fn submit<const QUEUED: bool>(
        replays: &Replays<Graph<QUEUED>>,
        runs: &[Shaped],
        budget: GraphCensus,
        made: impl Fn(&Graph<QUEUED>),
    ) -> Vec<Vec<u64>> {
        let taken = replays
            .record(runs, budget, &retained(), || Ok(()), |index| {
                let mut recorded = graph(4);
                recorded.1.clone_from(&runs[index].launch_bound);
                made(&recorded);
                Ok(recorded)
            })
            .unwrap();
        let mut bound = Vec::new();
        for (run, &identity) in runs.iter().zip(&taken) {
            replays
                .replay(run, identity, |graph, _| Ok(graph.1.clone_from(&run.launch_bound)), |graph| Ok(bound.push(graph.1.clone())))
                .unwrap();
        }
        bound
    }

    /// Where graphs are recorded at their first use (Vulkan), graphs no
    /// submission takes are released, the least recently replayed first,
    /// while the graphs exceed the budget.
    #[test]
    fn graphs_fit_the_budget_their_owners_hold() {
        let replays = Replays::<Graph<true>>::default();
        let budget = GraphCensus { graphs: 2, launches: 8, arguments: 128, largest: 16 };
        for key in 0..3 {
            submit(&replays, &[shaped(key, 0)], budget, |_| ());
        }
        let mut kept = replays.entries.lock().unwrap().graphs.keys().map(|key| key.to_vec()).collect::<Vec<_>>();
        kept.sort();
        assert_eq!(kept, vec![vec![1], vec![2]]);
    }

    /// Graphs kept for launches in flight (`Replays::ensure`) stay as other
    /// submissions release graphs beyond the budget, the oldest replayed
    /// first; once their launches end, they are released like any other.
    #[test]
    fn graphs_kept_for_launches_in_flight_stay_beyond_the_budget() {
        let replays = Replays::<Graph<false>>::default();
        let budget = GraphCensus { graphs: 2, launches: 8, arguments: 128, largest: 16 };
        let flights = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        replays.ensure(&[0], 1, &flights, budget, &retained(), || Ok(()), || Ok(graph(4))).unwrap();
        for key in 1..4 {
            submit(&replays, &[shaped(key, 0)], budget, |_| ());
        }
        let kept = |replays: &Replays<Graph<false>>| {
            let mut kept = replays.entries.lock().unwrap().graphs.keys().map(|key| key.to_vec()).collect::<Vec<_>>();
            kept.sort();
            kept
        };
        assert_eq!(kept(&replays), vec![vec![0], vec![3]], "the kept graph stays, the oldest of the rest go");
        flights.store(0, std::sync::atomic::Ordering::Release);
        submit(&replays, &[shaped(4, 0), shaped(5, 0)], budget, |_| ());
        assert_eq!(kept(&replays), vec![vec![4], vec![5]], "with no launch in flight it is released");
    }

    /// A shape two families run is kept for both families' launches in
    /// flight: one family's demand adds to the other's, and its launches
    /// ending releases only its own share.
    #[test]
    fn a_shape_two_families_run_is_kept_for_both() {
        let replays = Replays::<Graph<false>>::default();
        let budget = |graphs: u64| GraphCensus { graphs, launches: 4 * graphs, arguments: 64 * graphs, largest: 16 };
        let (first, second) = (Arc::new(std::sync::atomic::AtomicUsize::new(2)), Arc::new(std::sync::atomic::AtomicUsize::new(1)));
        replays.ensure(&[0], 1, &first, budget(4), &retained(), || Ok(()), || Ok(graph(4))).unwrap();
        replays.ensure(&[0], 1, &second, budget(4), &retained(), || Ok(()), || Ok(graph(4))).unwrap();
        // Again for the first family: its demand replaces its last, beside
        // the second's, rather than adding to it.
        replays.ensure(&[0], 1, &first, budget(4), &retained(), || Ok(()), || Ok(graph(4))).unwrap();
        let held = |replays: &Replays<Graph<false>>| replays.entries.lock().unwrap().graphs.get(&[0][..]).map_or(0, |replay| replay.graphs.len());
        assert_eq!(held(&replays), 3, "both families' launches in flight are kept");
        submit(&replays, &[shaped(1, 0)], budget(4), |_| ());
        submit(&replays, &[shaped(2, 0)], budget(4), |_| ());
        assert_eq!(held(&replays), 3, "a release beyond the budget takes neither family's");
        first.store(0, std::sync::atomic::Ordering::Release);
        submit(&replays, &[shaped(3, 0)], budget(3), |_| ());
        assert_eq!(held(&replays), 1, "the first family's launches ended, so only the second's stays");
    }

    /// The budget holds every graph its owners' runs in flight can need, so
    /// a submission whose own graphs exceed it breaks an invariant.
    #[test]
    #[should_panic(expected = "exceed every graph its owners hold")]
    fn a_submission_beyond_the_budget_is_an_invariant_violation() {
        let replays = Replays::<Graph<true>>::default();
        let budget = GraphCensus { graphs: 2, launches: 8, arguments: 128, largest: 16 };
        submit(&replays, &[shaped(0, 0), shaped(1, 0), shaped(2, 0)], budget, |_| ());
    }

    const PLACES: u64 = 7;
    const SHAPES: u64 = 3;

    /// A chain's runs: each shape at each place, bound to the place's storage.
    fn chain() -> Vec<Shaped> {
        (0..SHAPES).flat_map(|shape| (0..PLACES).map(move |place| shaped(shape, 100 * place + shape))).collect()
    }

    /// Where a graph is updated only once idle (CUDA), a chain over N places
    /// in flight together records a graph per shape and place at its first
    /// use and every later chain replays them as bound: no recording, no
    /// update, no wait.
    #[test]
    fn a_chain_in_flight_records_each_place_once_and_replays_it_as_bound() {
        let replays = Replays::<Graph<false>>::default();
        let budget = GraphCensus { graphs: SHAPES * PLACES, launches: 4 * SHAPES * PLACES, arguments: 64 * SHAPES * PLACES, largest: 16 };
        let chain = chain();
        let expected = chain.iter().map(|run| run.launch_bound.clone()).collect::<Vec<_>>();
        for _ in 0..3 {
            assert_eq!(submit(&replays, &chain, budget, |_| ()), expected);
        }
        assert_eq!(replays.counts(), GraphReplays { recorded: SHAPES * PLACES, updated: 0 });
    }

    /// Where updates run in queue order (Vulkan), one graph per shape serves
    /// every place of a chain, recorded once and updated at each place whose
    /// storage differs from what it was bound to.
    #[test]
    fn a_chain_updated_in_queue_order_records_each_shape_once() {
        let replays = Replays::<Graph<true>>::default();
        let budget = GraphCensus { graphs: SHAPES, launches: 4 * SHAPES, arguments: 64 * SHAPES, largest: 16 };
        let chain = chain();
        let expected = chain.iter().map(|run| run.launch_bound.clone()).collect::<Vec<_>>();
        for _ in 0..2 {
            assert_eq!(submit(&replays, &chain, budget, |_| ()), expected);
        }
        assert_eq!(
            replays.counts(),
            GraphReplays { recorded: SHAPES, updated: SHAPES * (PLACES - 1) + SHAPES * PLACES }
        );
    }

    /// A binding no graph holds takes an idle graph of its shape, updated,
    /// and records a new one while every graph of the shape is in flight.
    #[test]
    fn a_new_binding_updates_only_an_idle_graph() {
        let replays = Replays::<Graph<false>>::default();
        let budget = GraphCensus { graphs: 2, launches: 8, arguments: 128, largest: 16 };
        // The graph bound to 1 stays in flight.
        submit(&replays, &[shaped(0, 1)], budget, |graph| graph.2.store(false, std::sync::atomic::Ordering::Relaxed));
        // The graph bound to 1 is in flight: binding 2 records another.
        submit(&replays, &[shaped(0, 2)], budget, |_| ());
        assert_eq!(replays.counts(), GraphReplays { recorded: 2, updated: 0 });
        // Binding 3 takes the idle one, bound to 2, and updates it.
        assert_eq!(submit(&replays, &[shaped(0, 3)], budget, |_| ()), vec![vec![3]]);
        assert_eq!(replays.counts(), GraphReplays { recorded: 2, updated: 1 });
    }

    /// A run that lacks a graph has the device prepared for its recording
    /// first (CUDA's drain), once a submission, and only then: a run whose
    /// graph is kept, bound or idle, records nothing and prepares nothing.
    #[test]
    fn only_a_recording_prepares_the_device() {
        let replays = Replays::<Graph<false>>::default();
        let budget = GraphCensus { graphs: 2, launches: 8, arguments: 128, largest: 16 };
        let prepared = std::cell::Cell::new(0);
        let submit = |runs: &[Shaped]| {
            replays
                .record(runs, budget, &retained(), || Ok(prepared.set(prepared.get() + 1)), |_| Ok(graph(4)))
                .unwrap()
        };
        submit(&[shaped(0, 1), shaped(1, 1)]);
        assert_eq!(prepared.get(), 1, "one preparation records both");
        submit(&[shaped(0, 1), shaped(1, 2)]);
        assert_eq!(prepared.get(), 1, "a kept graph, bound or idle, prepares nothing");
        assert_eq!(replays.counts().recorded, 2);
    }

    /// An update whose queued work never ran (its submission failed) leaves
    /// no claim on what the graph binds: after `unbind`, or a failure inside
    /// the update itself, the next replay updates the graph again.
    #[test]
    fn a_failed_submission_leaves_its_graphs_updated_again() {
        let replays = Replays::<Graph<true>>::default();
        let budget = GraphCensus { graphs: 1, launches: 4, arguments: 64, largest: 16 };
        let (first, second) = (shaped(0, 1), shaped(0, 2));
        submit(&replays, std::slice::from_ref(&first), budget, |_| ());
        let taken = replays.record(std::slice::from_ref(&second), budget, &retained(), || unreachable!(), |_| unreachable!()).unwrap();
        replays.replay(&second, taken[0], |graph, _| Ok(graph.1.clone_from(&second.launch_bound)), |_| Ok(())).unwrap();
        replays.unbind(std::slice::from_ref(&second), |graph| graph.1.clear());
        let updates = std::cell::Cell::new(0);
        let count = |graph: &mut Graph<true>, run: &Shaped| {
            updates.set(updates.get() + 1);
            graph.1.clone_from(&run.launch_bound);
            Ok(())
        };
        replays.replay(&second, taken[0], |graph, _| count(graph, &second), |_| Ok(())).unwrap();
        assert_eq!(updates.get(), 1, "updated again after the failed submission");
        let failed = replays.replay(&first, taken[0], |_, _| Err(CallError::Workflow(crate::api::WorkflowError::Empty)), |_| Ok(()));
        assert!(failed.is_err());
        replays.replay(&first, taken[0], |graph, _| count(graph, &first), |_| Ok(())).unwrap();
        assert_eq!(updates.get(), 2, "updated again after its own failure");
    }

    /// `by_plan` entries are pruned as plans drop, including shared layout keys
    /// where the entry stays until the last plan sharing it drops.
    #[test]
    fn dropped_plans_are_pruned_from_by_plan() {
        let replays = Replays::<Graph<true>>::default();
        let budget = GraphCensus { graphs: 4, launches: 16, arguments: 256, largest: 16 };
        // Two plans sharing layout key 7 (CUDA case): dropping one keeps counts.
        replays.register_plan(7);
        replays.register_plan(7);
        submit(&replays, &[shaped(7, 1)], budget, |_| ());
        assert_eq!(replays.plan_counts(7).recorded, 1);
        replays.unregister_plan(7);
        assert_eq!(replays.plan_counts(7).recorded, 1, "still live while second plan exists");
        replays.unregister_plan(7);
        assert_eq!(replays.by_plan_len(), 0, "pruned once last plan for key drops");

        // Many sequential plans across engine lifetime leave zero retained entries.
        for plan in 100..600u64 {
            replays.register_plan(plan);
            submit(&replays, &[shaped(plan, 0)], budget, |_| ());
            replays.unregister_plan(plan);
        }
        assert_eq!(replays.by_plan_len(), 0, "by_plan stays bounded as plans churn");
    }

    /// Unbound host-only schedules (`with_unbound`, Metal) hit without re-deriving,
    /// do not pollute `by_plan` or `counts()`, stay bounded by LRU eviction, and
    /// clear completely when unheld.
    #[test]
    fn with_unbound_caches_without_thrashing_or_by_plan_leak() {
        let replays = Replays::<Graph<true>>::default();
        let budget = GraphCensus { graphs: 4, launches: 16, arguments: 256, largest: 16 };
        let derived = std::cell::Cell::new(0usize);
        let run = |key: &[u64]| {
            replays
                .with_unbound(
                    key,
                    budget,
                    || {
                        derived.set(derived.get() + 1);
                        graph(4)
                    },
                    |_| Ok(()),
                )
                .unwrap();
        };
        // Alternating across 4 active shapes fits within budget: derives once each.
        for _ in 0..50 {
            for shape in 0..4u64 {
                run(&[shape, 1, 2, 3, 4]);
            }
        }
        assert_eq!(derived.get(), 4, "no thrashing when working set fits budget");
        assert_eq!(replays.by_plan_len(), 0, "unbound schedules do not pollute by_plan");
        assert_eq!(replays.counts(), GraphReplays::default(), "unbound schedules are not counted");

        // Churning 200 distinct shapes stays bounded at budget.graphs.
        for shape in 10..210u64 {
            run(&[shape, 1, 2, 3, 4]);
        }
        assert_eq!(replays.entries.lock().unwrap().graphs.len(), 4);
        replays.clear_unheld(|| true);
        assert_eq!(replays.entries.lock().unwrap().graphs.len(), 0);
    }

    /// When a layout already has a graph in flight, `record_or_direct` skips
    /// `prepare` (no `device.drain()`) and returns `None` for direct launch;
    /// when a layout has no graph yet (or a submission needs more distinct
    /// instances than exist), it calls `prepare` and records normally.
    #[test]
    fn record_or_direct_skips_prepare_when_existing_graphs_are_in_flight() {
        let replays = Replays::<Graph<false>>::default();
        let budget = GraphCensus { graphs: 4, launches: 16, arguments: 256, largest: 16 };
        let prepared = std::cell::Cell::new(0);
        let in_flight = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = in_flight.clone();

        // Cold start: shape 0 has no graph yet -> calls prepare and records 1 graph.
        let first = replays
            .record_or_direct(
                &[shaped(0, 1)],
                budget,
                &retained(),
                || {
                    prepared.set(prepared.get() + 1);
                    Ok(())
                },
                |_| {
                    let mut g = graph(4);
                    g.1 = vec![1];
                    g.2 = flag.clone();
                    Ok(g)
                },
            )
            .unwrap();
        assert_eq!(prepared.get(), 1);
        assert_eq!(first.len(), 1);
        assert!(first[0].is_some());
        assert_eq!(replays.counts().recorded, 1);

        // Warm prefill: shape 0's graph is in flight (`idle == false`), different
        // binding arrives -> skips prepare (0 drains!), returns None for direct launch.
        let busy = replays
            .record_or_direct(
                &[shaped(0, 2)],
                budget,
                &retained(),
                || panic!("must not call prepare when an existing graph is merely in flight"),
                |_| panic!("must not record a new graph when falling back to direct launch"),
            )
            .unwrap();
        assert_eq!(busy, vec![None]);
        assert_eq!(prepared.get(), 1);
        assert_eq!(replays.counts().recorded, 1);

        // Once the graph becomes idle, a new binding reuses it without prepare.
        in_flight.store(true, std::sync::atomic::Ordering::Relaxed);
        let reused = replays
            .record_or_direct(
                &[shaped(0, 3)],
                budget,
                &retained(),
                || panic!("must not call prepare when an idle graph is available"),
                |_| panic!("must not record when an idle graph is available"),
            )
            .unwrap();
        assert_eq!(reused, first);
        assert_eq!(prepared.get(), 1);

        // A single submission needing 2 distinct instances of shape 0 when only 1
        // exists genuinely needs a 2nd graph -> calls prepare and records it.
        let chain = replays
            .record_or_direct(
                &[shaped(0, 10), shaped(0, 11)],
                budget,
                &retained(),
                || {
                    prepared.set(prepared.get() + 1);
                    Ok(())
                },
                |_| Ok(graph(4)),
            )
            .unwrap();
        assert_eq!(prepared.get(), 2);
        assert!(chain[0].is_some() && chain[1].is_some() && chain[0] != chain[1]);
        assert_eq!(replays.counts().recorded, 2);
    }

    #[test]
    fn record_or_direct_evicts_and_falls_back_to_direct_on_capacity_refusal() {
        use seismic_compiler::errors::{ExecutionError, RefusalReading};

        let replays = Replays::<Graph<false>>::default();
        let budget = GraphCensus { graphs: 8, launches: 32, arguments: 512, largest: 32 };
        let capacity_err = || {
            CallError::Execution(ExecutionError::DeviceRefused {
                required: 2 * 1024 * 1024,
                reading: RefusalReading::DeviceFree { free: 2_686_976, total: 33_711_521_792 },
            })
        };

        // Seed shape 0 (e.g. warmup decode graph) into the cache.
        let warmup = replays
            .record_or_direct(&[shaped(0, 1)], budget, &retained(), || Ok(()), |_| Ok(graph(4)))
            .unwrap();
        assert!(warmup[0].is_some());
        assert_eq!(replays.counts().recorded, 1);

        // New shape 1 hits DeviceRefused on first attempt; `record_or_direct`
        // evicts untaken shape 0 and retries once, succeeding on the retry.
        let attempts = std::cell::Cell::new(0usize);
        let recovered = replays
            .record_or_direct(
                &[shaped(1, 10)],
                budget,
                &retained(),
                || Ok(()),
                |_| {
                    let n = attempts.get();
                    attempts.set(n + 1);
                    if n == 0 {
                        Err(capacity_err())
                    } else {
                        Ok(graph(4))
                    }
                },
            )
            .unwrap();
        assert_eq!(attempts.get(), 2, "must retry once after evicting untaken graphs");
        assert!(recovered[0].is_some(), "retry after eviction succeeds");

        // Now a 3-run submission for shape 2 hits persistent DeviceRefused even
        // after evicting shape 1: falls back to `None` (direct launch) for all
        // unrecorded runs without error, and short-circuits remaining runs after
        // capacity is exhausted.
        let oom_attempts = std::cell::Cell::new(0usize);
        let fallback = replays
            .record_or_direct(
                &[shaped(2, 20), shaped(2, 21), shaped(2, 22)],
                budget,
                &retained(),
                || Ok(()),
                |_| {
                    oom_attempts.set(oom_attempts.get() + 1);
                    Err(capacity_err())
                },
            )
            .unwrap();
        assert_eq!(fallback, vec![None, None, None]);
        assert_eq!(
            oom_attempts.get(),
            2,
            "first run tries + evicts shape 1 + retries once, then remaining runs short-circuit to Direct"
        );
    }

    #[test]
    fn record_or_direct_propagates_non_capacity_errors() {
        use seismic_compiler::errors::ExecutionError;

        let replays = Replays::<Graph<false>>::default();
        let budget = GraphCensus { graphs: 4, launches: 16, arguments: 256, largest: 16 };
        let err = replays
            .record_or_direct(
                &[shaped(0, 1)],
                budget,
                &retained(),
                || Ok(()),
                |_| Err(CallError::Execution(ExecutionError::SubmissionFailed("driver fault".into()))),
            )
            .unwrap_err();
        assert!(matches!(err, CallError::Execution(ExecutionError::SubmissionFailed(_))));
    }

    #[test]
    fn work_bound_release_beyond_o1_and_choose_linear() {
        let budget = GraphCensus {
            graphs: 128,
            launches: 1024,
            arguments: 16 * 1024,
            largest: 64,
        };
        let measure_census_on_hit = |unrelated_shapes: u64| -> usize {
            let replays = Replays::<Graph<false>>::default();
            for key in 0..unrelated_shapes {
                replays
                    .record(&[shaped(key, 1)], budget, &retained(), || Ok(()), |_| Ok(graph(4)))
                    .unwrap();
            }
            CENSUS_CALLS.with(|c| c.set(0));
            replays
                .record(&[shaped(0, 1)], budget, &retained(), || Ok(()), |_| unreachable!("cached"))
                .unwrap();
            CENSUS_CALLS.with(|c| c.get())
        };

        // Steady-state hit within budget must do O(1) census work (0 Graph::census
        // calls) regardless of 1 vs 64 unrelated cached graphs.
        assert_eq!(measure_census_on_hit(1), 0);
        assert_eq!(measure_census_on_hit(64), 0);

        // `choose()` on R=64 unmatched runs sharing 1 shape and 1 binding must
        // resolve in O(R) with O(1) idle checks (not O(R^2)).
        let replays = Replays::<Graph<false>>::default();
        replays
            .record(&[shaped(10, 1)], budget, &retained(), || Ok(()), |_| Ok(graph(4)))
            .unwrap();
        let runs: Vec<Shaped> = (0..64).map(|_| shaped(10, 2)).collect();
        IDLE_CALLS.with(|c| c.set(0));
        CENSUS_CALLS.with(|c| c.set(0));
        let taken = replays
            .record(&runs, budget, &retained(), || Ok(()), |_| unreachable!("reused"))
            .unwrap();
        assert_eq!(taken.len(), 64);
        assert!(taken.iter().all(|&id| id == taken[0]));
        assert_eq!(CENSUS_CALLS.with(|c| c.get()), 0);
        assert_eq!(
            IDLE_CALLS.with(|c| c.get()),
            1,
            "64 identical unmatched runs must query idle() once, not O(R^2)"
        );
    }

    /// Monotonic timeline-word value changes (`word_bound` incrementing every
    /// step while `launch_bound` stays identical across 2xN pipelined graphs)
    /// must hit Tier 2 (`launch_bound` match -> `ReplayUpdate::WordsOnly`, zero
    /// `bind_launches` and zero `updated` increments) instead of falling
    /// through to LRU and thrashing launch bindings across layers/slots.
    #[test]
    fn monotonic_word_bound_hits_tier2_without_rebinding_launches() {
        const PLACES_N: u64 = 4;
        const SLOTS: u64 = 2;
        let replays = Replays::<Graph<false>>::default();
        let total = SLOTS * PLACES_N;
        let budget = GraphCensus {
            graphs: total,
            launches: 4 * total,
            arguments: 64 * total,
            largest: 16,
        };
        let slot_idle = [
            Arc::new(std::sync::atomic::AtomicBool::new(true)),
            Arc::new(std::sync::atomic::AtomicBool::new(true)),
        ];

        let mut recorded_ids: [Vec<u64>; 2] = [Vec::new(), Vec::new()];
        let run_slot = |slot: usize, step: u64| -> (Vec<u64>, usize, usize) {
            let runs: Vec<Shaped> = (0..PLACES_N)
                .map(|place| shaped_with_word(0, 100 * (slot as u64) + place, 0xBEEF, step * 100 + place))
                .collect();
            let flag = slot_idle[slot].clone();
            let taken = replays
                .record(&runs, budget, &retained(), || Ok(()), |index| {
                    let mut g = graph(4);
                    g.1.clone_from(&runs[index].launch_bound);
                    g.2 = flag.clone();
                    Ok(g)
                })
                .unwrap();
            let mut launch_updates = 0usize;
            let mut word_updates = 0usize;
            for (run, &identity) in runs.iter().zip(&taken) {
                replays
                    .replay(
                        run,
                        identity,
                        |g, kind| {
                            match kind {
                                ReplayUpdate::Launches => {
                                    launch_updates += 1;
                                    g.1.clone_from(&run.launch_bound);
                                }
                                ReplayUpdate::WordsOnly => {
                                    word_updates += 1;
                                }
                            }
                            Ok(())
                        },
                        |_| Ok(()),
                    )
                    .unwrap();
            }
            (taken, launch_updates, word_updates)
        };

        // Cold step 0: submit slot 0 (stays in flight), then slot 1 -> records 2*N graphs.
        slot_idle[0].store(false, std::sync::atomic::Ordering::Relaxed);
        let (ids0, lu0, wu0) = run_slot(0, 1);
        assert_eq!((lu0, wu0), (0, 0), "recorded_bound=true binds on record");
        slot_idle[1].store(false, std::sync::atomic::Ordering::Relaxed);
        let (ids1, lu1, wu1) = run_slot(1, 2);
        assert_eq!((lu1, wu1), (0, 0));
        recorded_ids[0] = ids0;
        recorded_ids[1] = ids1;
        assert_eq!(replays.counts(), GraphReplays { recorded: total, updated: 0 });

        // Steady-state pipelined decode (steps 3..12): each slot completes before
        // its next turn while the other slot remains in flight. `word_bound`
        // increments monotonically every step; `launch_bound` stays identical.
        for step in 3..13u64 {
            let slot = (step as usize) % 2;
            slot_idle[slot].store(true, std::sync::atomic::Ordering::Relaxed);
            let (ids, launch_updates, word_updates) = run_slot(slot, step);
            slot_idle[slot].store(false, std::sync::atomic::Ordering::Relaxed);
            assert_eq!(ids, recorded_ids[slot], "step {step} slot {slot} must keep its exact graph identities");
            assert_eq!(launch_updates, 0, "step {step} must hit Tier 2 with zero bind_launches");
            assert_eq!(word_updates, PLACES_N as usize, "step {step} must update only timeline words");
        }
        assert_eq!(
            replays.counts(),
            GraphReplays { recorded: total, updated: 0 },
            "steady-state monotonic word updates must record 0 graphs and 0 launch rebinds"
        );

        // Tier 1 exact match: replaying the exact same `(launch_bound, word_bound)`
        // once idle invokes neither launch nor word updates.
        slot_idle[0].store(true, std::sync::atomic::Ordering::Relaxed);
        let (ids_exact, launch_exact, word_exact) = run_slot(0, 12);
        assert_eq!(ids_exact, recorded_ids[0]);
        assert_eq!((launch_exact, word_exact), (0, 0), "exact match hits Tier 1 with zero updates");
    }

    /// Capacity refusal during `record_or_direct` must evict only the oldest
    /// half of untaken cached graphs first so recently-used hot graphs survive,
    /// falling back to evicting the remaining half only if the retry still
    /// refuses.
    #[test]
    fn record_or_direct_evicts_incrementally_preserving_hot_graphs_on_capacity_refusal() {
        use seismic_compiler::errors::{ExecutionError, RefusalReading};

        let replays = Replays::<Graph<false>>::default();
        let budget = GraphCensus { graphs: 16, launches: 64, arguments: 1024, largest: 32 };
        let capacity_err = || {
            CallError::Execution(ExecutionError::DeviceRefused {
                required: 2 * 1024 * 1024,
                reading: RefusalReading::DeviceFree { free: 1_048_576, total: 24 * 1024 * 1024 * 1024 },
            })
        };

        // Seed 6 distinct shapes (0..6): shapes 0..3 are oldest, shapes 3..6 are hottest.
        let mut seeded_ids = Vec::new();
        for shape in 0..6u64 {
            let ids = replays
                .record_or_direct(&[shaped(shape, shape + 1)], budget, &retained(), || Ok(()), |_| Ok(graph(4)))
                .unwrap();
            seeded_ids.push(ids[0].unwrap());
        }
        assert_eq!(replays.counts().recorded, 6);

        // Recording shape 10 hits DeviceRefused once, then succeeds after the first
        // half-eviction (evicting oldest 3 shapes 0, 1, 2 while keeping hot 3, 4, 5).
        let attempts = std::cell::Cell::new(0usize);
        let res = replays
            .record_or_direct(
                &[shaped(10, 100)],
                budget,
                &retained(),
                || Ok(()),
                |_| {
                    let n = attempts.get();
                    attempts.set(n + 1);
                    if n == 0 { Err(capacity_err()) } else { Ok(graph(4)) }
                },
            )
            .unwrap();
        assert_eq!(attempts.get(), 2);
        assert!(res[0].is_some());
        assert_eq!(replays.counts().recorded, 7);

        // Hottest shapes (3, 4, 5) must still hit the cache with zero new recordings.
        for shape in 3..6u64 {
            let ids = replays
                .record_or_direct(&[shaped(shape, shape + 1)], budget, &retained(), || Ok(()), |_| {
                    panic!("hot shape {shape} should have survived incremental eviction")
                })
                .unwrap();
            assert_eq!(ids[0], Some(seeded_ids[shape as usize]));
        }
        assert_eq!(replays.counts().recorded, 7);
    }
}
