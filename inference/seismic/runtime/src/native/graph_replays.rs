//! A device's recorded graphs of sealed-plan submissions (CUDA graphs,
//! Vulkan secondary command buffers), keyed by the plans' identities
//! followed by every buffer address the submission binds: with the sealed
//! words and geometry they fix every launch argument. A submission binding
//! the same storage as an earlier one replays that submission's graph; one
//! binding new storage records a new graph.

use crate::api::CallError;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Graphs one device keeps; the least recently replayed is dropped beyond
/// it. The steady state is one graph per step shape (row class and batch)
/// and workspace slot, plus the runs submitted alone.
const REPLAY_CAPACITY: usize = 1024;

/// The most device memory one device's recorded graphs hold (CUDA graph
/// instantiation takes driver memory, 2 to 4 MiB a graph under WDDM; a
/// Vulkan graph's secondary command buffers take command memory); the least
/// recently replayed graphs are dropped beyond it. Assessment plans it as a
/// fixed term of every CUDA and Vulkan load
/// ([`super::recorded_graph_device_bytes`]).
pub(super) const GRAPH_DEVICE_BYTES: u64 = 192 << 20;

/// A recorded graph and the device memory it holds.
pub(crate) trait RecordedGraph {
    fn device_bytes(&self) -> u64;
}

pub(crate) struct Replays<G> {
    entries: Mutex<ReplayEntries<G>>,
}

impl<G> Default for Replays<G> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(ReplayEntries {
                clock: 0,
                graphs: HashMap::new(),
            }),
        }
    }
}

struct ReplayEntries<G> {
    clock: u64,
    graphs: HashMap<Vec<u64>, Replay<G>>,
}

struct Replay<G> {
    /// Clock of the last replay.
    used: u64,
    graph: G,
    /// The plans the graph's kernels belong to, kept loaded while it lives.
    _retained: Arc<dyn Any + Send + Sync>,
}

impl<G: RecordedGraph> Replays<G> {
    /// Run `replay` with the graph of `key`, recording it with `record` when
    /// it is absent (dropping the least recently replayed graph at
    /// capacity).
    pub(crate) fn replay(
        &self,
        key: Vec<u64>,
        retained: &Arc<dyn Any + Send + Sync>,
        record: impl FnOnce() -> Result<G, CallError>,
        replay: impl FnOnce(&mut G) -> Result<(), CallError>,
    ) -> Result<(), CallError> {
        let mut entries = self
            .entries
            .lock()
            .expect("native graph replays lock is never poisoned");
        entries.clock += 1;
        let clock = entries.clock;
        if !entries.graphs.contains_key(&key) {
            if entries.graphs.len() == REPLAY_CAPACITY {
                let oldest = entries
                    .graphs
                    .iter()
                    .min_by_key(|(_, replay)| replay.used)
                    .map(|(key, _)| key.clone())
                    .expect("a full replay set has an entry");
                entries.graphs.remove(&oldest);
            }
            let graph = record()?;
            // Drop the least recently replayed graphs until the new one fits
            // the device memory graphs may hold.
            let mut held = entries
                .graphs
                .values()
                .map(|replay| replay.graph.device_bytes())
                .sum::<u64>()
                + graph.device_bytes();
            while held > GRAPH_DEVICE_BYTES && !entries.graphs.is_empty() {
                let oldest = entries
                    .graphs
                    .iter()
                    .min_by_key(|(_, replay)| replay.used)
                    .map(|(key, _)| key.clone())
                    .expect("a non-empty replay set has an entry");
                held -= entries
                    .graphs
                    .remove(&oldest)
                    .expect("the oldest graph is recorded")
                    .graph
                    .device_bytes();
            }
            entries.graphs.insert(
                key.clone(),
                Replay {
                    used: clock,
                    graph,
                    _retained: retained.clone(),
                },
            );
        }
        let entry = entries
            .graphs
            .get_mut(&key)
            .expect("the submission's graph was recorded above");
        entry.used = clock;
        replay(&mut entry.graph)
    }
}
