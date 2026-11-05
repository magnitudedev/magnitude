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
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    /// beside other loads on the device.
    by_plan: Mutex<HashMap<u64, GraphReplays>>,
}

impl<G> Default for Replays<G> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(ReplayEntries {
                clock: 0,
                next: 0,
                graphs: HashMap::new(),
            }),
            recorded: std::sync::atomic::AtomicU64::new(0),
            updated: std::sync::atomic::AtomicU64::new(0),
            by_plan: Mutex::new(HashMap::new()),
        }
    }
}

/// One graph a submission replays: its shape's key, and the parameters it
/// binds there (every buffer address its launches take, on a backend whose
/// graphs hold them; none on one whose graphs hold no parameter).
pub(crate) struct Shaped {
    pub(crate) key: Vec<u64>,
    pub(crate) bound: Vec<u64>,
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
    graphs: HashMap<Vec<u64>, Replay<G>>,
}

/// A shape's graphs.
struct Replay<G> {
    graphs: Vec<Instance<G>>,
    /// The plans the graphs' kernels belong to, kept loaded while they live.
    _retained: Arc<dyn Any + Send + Sync>,
    /// The graphs of it kept for each launch in flight ([`Replays::ensure`]).
    pinned: Option<Pinned>,
}

/// Graphs of a shape kept whatever the budget: `per_flight` for each of
/// `flights`, a family's live slots, each a launch in flight. The family's
/// hold counts them (`LaunchGraphs::held`: a decode launch's graphs for
/// each launch beside one, and within the union for that one), so a
/// release never needs them.
struct Pinned {
    per_flight: usize,
    flights: Arc<std::sync::atomic::AtomicUsize>,
}

impl<G> Replay<G> {
    /// The graphs of it no release takes.
    fn pinned(&self) -> usize {
        self.pinned.as_ref().map_or(0, |pinned| pinned.per_flight * pinned.flights.load(std::sync::atomic::Ordering::Acquire))
    }
}

struct Instance<G> {
    /// Its identity, which a submission's replays name it by
    /// ([`Replays::record`]).
    identity: u64,
    graph: G,
    /// The parameters it binds once the work queued so far ran; none where
    /// that is unknown (`Replays::unbind`).
    bound: Option<Vec<u64>>,
    /// Clock of its last replay.
    used: u64,
}

impl<G> Replays<G> {
    /// What the graphs cost serving so far.
    pub(crate) fn counts(&self) -> GraphReplays {
        GraphReplays {
            recorded: self.recorded.load(std::sync::atomic::Ordering::Relaxed),
            updated: self.updated.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// What the graphs of plan `plan`'s runs cost so far.
    pub(crate) fn plan_counts(&self, plan: u64) -> GraphReplays {
        self.by_plan.lock().expect("graph counts lock is never poisoned").get(&plan).copied().unwrap_or_default()
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
        count(self.by_plan.lock().expect("graph counts lock is never poisoned").entry(plan).or_default());
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
    /// invariant. Returns each run's graph's identity, which
    /// [`Self::replay`] takes.
    pub(crate) fn record(
        &self,
        graphs: &[Shaped],
        budget: GraphCensus,
        retained: &Arc<dyn Any + Send + Sync>,
        prepare: impl FnOnce() -> Result<(), CallError>,
        mut record: impl FnMut(usize) -> Result<G, CallError>,
    ) -> Result<Vec<u64>, CallError> {
        let lock = || self.entries.lock().expect("native graph replays lock is never poisoned");
        let mut entries = lock();
        let mut taken = Self::choose(&entries, graphs)?;
        if taken.contains(&None) {
            drop(entries);
            prepare()?;
            entries = lock();
            taken = Self::choose(&entries, graphs)?;
        }
        // The rest record a graph each, which every later run of that
        // binding takes, and where updates run in queue order every later
        // run of its shape.
        for index in 0..graphs.len() {
            if taken[index].is_some() {
                continue;
            }
            let Shaped { key, bound } = &graphs[index];
            let graph = record(index)?;
            let identity = self.keep(&mut entries, key, graph, G::recorded_bound().then(|| bound.clone()), retained);
            for later in index..graphs.len() {
                if taken[later].is_none()
                    && graphs[later].key == *key
                    && (G::updates_in_flight() || graphs[later].bound == *bound)
                {
                    taken[later] = Some(identity);
                }
            }
        }
        let taken = taken.into_iter().map(|taken| taken.expect("every run takes a graph")).collect::<Vec<_>>();
        Self::fit(&mut entries, &taken, budget);
        Ok(taken)
    }

    /// Keep `per_flight` graphs of shape `key` for each of `flights`, those
    /// a launch of `per_flight` runs of it takes for each launch in flight,
    /// recorded ahead of them (`record`, after `prepare`, which runs only
    /// where one is to be recorded), each bound to nothing until its first
    /// replay; no release takes them while `flights` stands ([`Pinned`]).
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
        let lock = || self.entries.lock().expect("native graph replays lock is never poisoned");
        let kept = |entries: &ReplayEntries<G>| entries.graphs.get(key).map_or(0, |replay| replay.graphs.len());
        let count = per_flight * flights.load(std::sync::atomic::Ordering::Acquire);
        let mut entries = lock();
        if kept(&entries) < count {
            drop(entries);
            prepare()?;
            entries = lock();
        }
        for _ in kept(&entries)..count {
            let graph = record()?;
            self.keep(&mut entries, key, graph, None, retained);
        }
        if let Some(replay) = entries.graphs.get_mut(key) {
            replay.pinned = Some(Pinned { per_flight, flights: flights.clone() });
        }
        let own = entries.graphs.get(key).map(|replay| replay.graphs.iter().map(|instance| instance.identity).collect::<Vec<_>>()).unwrap_or_default();
        Self::fit(&mut entries, &own, budget);
        Ok(())
    }

    /// Keep `graph`, just recorded, of shape `key`, binding `bound` (none
    /// until its first replay binds it): counted recorded, and the most
    /// recently used, so it is released after every graph kept before it.
    /// Returns its identity.
    fn keep(
        &self,
        entries: &mut ReplayEntries<G>,
        key: &[u64],
        graph: G,
        bound: Option<Vec<u64>>,
        retained: &Arc<dyn Any + Send + Sync>,
    ) -> u64 {
        self.count_recorded(key);
        let identity = entries.next;
        entries.next += 1;
        entries.clock += 1;
        let used = entries.clock;
        entries
            .graphs
            .entry(key.to_vec())
            .or_insert_with(|| Replay { graphs: Vec::new(), _retained: retained.clone(), pinned: None })
            .graphs
            .push(Instance { identity, graph, bound, used });
        identity
    }

    /// The graph each of `graphs` takes of those kept ([`Self::record`]),
    /// none where it would record one.
    fn choose(entries: &ReplayEntries<G>, graphs: &[Shaped]) -> Result<Vec<Option<u64>>, CallError> {
        let mut taken: Vec<Option<u64>> = vec![None; graphs.len()];
        // A run whose binding a graph holds takes it, every run of that
        // binding alike.
        for (index, shaped) in graphs.iter().enumerate() {
            if let Some(replay) = entries.graphs.get(&shaped.key) {
                taken[index] = replay
                    .graphs
                    .iter()
                    .find(|instance| instance.bound.as_ref() == Some(&shaped.bound))
                    .map(|instance| instance.identity);
            }
        }
        // Another takes a graph of its shape that may be updated now: one no
        // other run takes, unless updates run in queue order.
        for index in 0..graphs.len() {
            if taken[index].is_some() {
                continue;
            }
            let Some(replay) = entries.graphs.get(&graphs[index].key) else {
                continue;
            };
            let mut free = None;
            for instance in &replay.graphs {
                let updatable = G::updates_in_flight()
                    || (!taken.contains(&Some(instance.identity)) && instance.graph.idle()?);
                if updatable && free.is_none_or(|(used, _)| instance.used < used) {
                    free = Some((instance.used, instance.identity));
                }
            }
            let free = free.map(|(_, identity)| identity);
            taken[index] = free;
            // The graph serves every later run of its binding, and where
            // updates run in queue order every other binding of its shape.
            for later in index + 1..graphs.len() {
                if taken[later].is_none()
                    && graphs[later].key == graphs[index].key
                    && (G::updates_in_flight() || graphs[later].bound == graphs[index].bound)
                {
                    taken[later] = free;
                }
            }
        }
        Ok(taken)
    }

    /// Release graphs while they exceed `budget`, the least recently
    /// replayed first: as a hold of it shrinks, so the graphs never hold
    /// more than their owners do.
    pub(crate) fn fit_to(&self, budget: GraphCensus) {
        let released = {
            let mut entries = self.entries.lock().expect("native graph replays lock is never poisoned");
            Self::release_beyond(&mut entries, &[], budget)
        };
        // Dropped after the lock, as what they keep may drop holds.
        drop(released);
    }

    /// Run `replay` with graph `identity` of `shaped`'s shape, which
    /// [`Self::record`] chose for this run, having first run `update` on it
    /// where it binds other parameters than `shaped`'s, given those it binds
    /// where known: on CUDA the changed launches' parameters set and their
    /// enable bits, and its words, on Vulkan their argument blocks rewritten
    /// in queue order.
    pub(crate) fn replay(
        &self,
        shaped: &Shaped,
        identity: u64,
        update: impl FnOnce(&mut G, Option<&[u64]>) -> Result<(), CallError>,
        replay: impl FnOnce(&mut G) -> Result<(), CallError>,
    ) -> Result<(), CallError> {
        let mut entries = self
            .entries
            .lock()
            .expect("native graph replays lock is never poisoned");
        entries.clock += 1;
        let clock = entries.clock;
        let instance = entries
            .graphs
            .get_mut(&shaped.key)
            .and_then(|replay| replay.graphs.iter_mut().find(|instance| instance.identity == identity))
            .expect("a submission replays only graphs chosen for it");
        instance.used = clock;
        if instance.bound.as_ref() != Some(&shaped.bound) {
            // Unknown until the update is queued: a failure in it leaves the
            // graph to be updated whole.
            let bound = instance.bound.take();
            update(&mut instance.graph, bound.as_deref())?;
            instance.bound = Some(shaped.bound.clone());
            self.count_updated(&shaped.key);
        }
        replay(&mut instance.graph)
    }

    /// Drop every graph if `unheld` says, under this lock, that no hold of
    /// the device's graph budget is live (`GraphHold::drop`): none can then
    /// be replayed, and each keeps its plans, which hold the device. A hold
    /// taken after the check records only once this lock is released. The
    /// graphs drop after it is, as what they keep may drop holds.
    pub(crate) fn clear_unheld(&self, unheld: impl FnOnce() -> bool) {
        let graphs = {
            let mut entries = self
                .entries
                .lock()
                .expect("native graph replays lock is never poisoned");
            if !unheld() {
                return;
            }
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
        let mut entries = self
            .entries
            .lock()
            .expect("native graph replays lock is never poisoned");
        for shaped in graphs {
            if let Some(replay) = entries.graphs.get_mut(&shaped.key) {
                for instance in &mut replay.graphs {
                    instance.bound = None;
                    forget(&mut instance.graph);
                }
            }
        }
    }

    /// Fit the graphs to `budget` now that the submission taking `taken` has
    /// replayed: a graph's census can change as it is replayed (a Vulkan
    /// division recorded again, `vulkan::Divided`).
    pub(crate) fn fit_after_replay(&self, taken: &[u64], budget: GraphCensus) {
        let mut entries = self
            .entries
            .lock()
            .expect("native graph replays lock is never poisoned");
        Self::fit(&mut entries, taken, budget);
    }

    /// Release graphs no run of the submission takes (`taken`), the least
    /// recently replayed first, while the graphs exceed `budget`. The budget
    /// holds every graph its owners' runs in flight can need, so the submission's
    /// own past it break an invariant.
    fn fit(entries: &mut ReplayEntries<G>, taken: &[u64], budget: GraphCensus) {
        let (released, held) = Self::release_beyond(entries, taken, budget);
        assert!(
            held.within(budget),
            "a submission's graphs, {held:?}, exceed every graph its owners hold, {budget:?}"
        );
        drop(released);
    }

    /// Take out graphs no run of a submission takes (`taken`), the least
    /// recently replayed first, while the graphs exceed `budget`; returns
    /// them, to be dropped, and the census of those kept.
    fn release_beyond(entries: &mut ReplayEntries<G>, taken: &[u64], budget: GraphCensus) -> (Vec<Instance<G>>, GraphCensus) {
        let census = |entries: &ReplayEntries<G>| {
            entries
                .graphs
                .values()
                .flat_map(|replay| &replay.graphs)
                .try_fold(GraphCensus::default(), |census, instance| census.plus(instance.graph.census()))
                .expect("a device's recorded graphs' census fits u64")
        };
        let mut held = census(entries);
        let mut released = Vec::new();
        if held.within(budget) {
            return (released, held);
        }
        let mut others = entries
            .graphs
            .iter()
            .flat_map(|(key, replay)| replay.graphs.iter().map(move |instance| (instance.used, instance.identity, key.clone())))
            .filter(|(_, identity, _)| !taken.contains(identity))
            .collect::<Vec<_>>();
        others.sort_unstable();
        for (_, identity, key) in others {
            let replay = entries.graphs.get_mut(&key).expect("an other graph's shape is recorded");
            // A shape's graphs kept for its launches in flight stay.
            if replay.graphs.len() <= replay.pinned() {
                continue;
            }
            let at = replay.graphs.iter().position(|instance| instance.identity == identity).expect("an other graph is kept");
            let instance = replay.graphs.swap_remove(at);
            held = held.minus(instance.graph.census()).expect("a kept graph is part of the census");
            released.push(instance);
            if replay.graphs.is_empty() {
                entries.graphs.remove(&key);
            }
            if held.within(budget) {
                break;
            }
        }
        (released, held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph of a census, the parameters it was last bound to, and whether
    /// its launches completed; `QUEUED` whether its updates run in queue
    /// order (Vulkan) rather than only when it is idle (CUDA).
    struct Graph<const QUEUED: bool>(GraphCensus, Vec<u64>, Arc<std::sync::atomic::AtomicBool>);

    impl<const QUEUED: bool> RecordedGraph for Graph<QUEUED> {
        fn census(&self) -> GraphCensus {
            self.0
        }
        fn idle(&self) -> Result<bool, CallError> {
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
        Shaped { key: vec![key], bound: vec![bound] }
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
                recorded.1.clone_from(&runs[index].bound);
                made(&recorded);
                Ok(recorded)
            })
            .unwrap();
        let mut bound = Vec::new();
        for (run, &identity) in runs.iter().zip(&taken) {
            replays
                .replay(run, identity, |graph, _| Ok(graph.1.clone_from(&run.bound)), |graph| Ok(bound.push(graph.1.clone())))
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
        let mut kept = replays.entries.lock().unwrap().graphs.keys().cloned().collect::<Vec<_>>();
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
            let mut kept = replays.entries.lock().unwrap().graphs.keys().cloned().collect::<Vec<_>>();
            kept.sort();
            kept
        };
        assert_eq!(kept(&replays), vec![vec![0], vec![3]], "the kept graph stays, the oldest of the rest go");
        flights.store(0, std::sync::atomic::Ordering::Release);
        submit(&replays, &[shaped(4, 0), shaped(5, 0)], budget, |_| ());
        assert_eq!(kept(&replays), vec![vec![4], vec![5]], "with no launch in flight it is released");
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
        let expected = chain.iter().map(|run| run.bound.clone()).collect::<Vec<_>>();
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
        let expected = chain.iter().map(|run| run.bound.clone()).collect::<Vec<_>>();
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
        replays.replay(&second, taken[0], |graph, _| Ok(graph.1.clone_from(&second.bound)), |_| Ok(())).unwrap();
        replays.unbind(std::slice::from_ref(&second), |graph| graph.1.clear());
        let updates = std::cell::Cell::new(0);
        let count = |graph: &mut Graph<true>, run: &Shaped| {
            updates.set(updates.get() + 1);
            graph.1.clone_from(&run.bound);
            Ok(())
        };
        replays.replay(&second, taken[0], |graph, _| count(graph, &second), |_| Ok(())).unwrap();
        assert_eq!(updates.get(), 1, "updated again after the failed submission");
        let failed = replays.replay(&first, taken[0], |_, _| Err(CallError::Workflow(crate::api::WorkflowError::Empty)), |_| Ok(()));
        assert!(failed.is_err());
        replays.replay(&first, taken[0], |graph, _| count(graph, &first), |_| Ok(())).unwrap();
        assert_eq!(updates.get(), 2, "updated again after its own failure");
    }
}
