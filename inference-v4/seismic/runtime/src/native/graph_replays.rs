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

impl<G> Replays<G> {
    /// Run `replay` with the graph of `key`, recording it with `record` when
    /// it is absent (dropping the least recently replayed graph at
    /// capacity).
    pub(crate) fn replay(
        &self,
        key: Vec<u64>,
        retained: &Arc<dyn Any + Send + Sync>,
        record: impl FnOnce() -> Result<G, CallError>,
        replay: impl FnOnce(&G) -> Result<(), CallError>,
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
        replay(&entry.graph)
    }
}
