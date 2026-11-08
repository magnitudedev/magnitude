//! What a device's recorded graphs hold in memory beyond its allocations,
//! as it follows from what the graphs are made of.
//!
//! A backend that replays sealed submissions records each as a graph: a
//! CUDA graph instantiated by the driver, Vulkan secondary command buffers
//! with their launches' argument blocks, a Metal schedule. What a set of
//! graphs is made of is its [`GraphCensus`]; what a device holds for a
//! census is its [`GraphCost`], which the backend measures once per device
//! where the memory is the driver's ([`GraphCost::Pooled`],
//! [`GraphCost::Chunked`]) and states as none where recording holds no
//! device memory.

use seismic_lang::registry::BackendName;

/// The argument bytes one launch's arguments hold in a recorded graph's
/// memory on `backend`, of `buffers` bound buffers and `words` bytes of
/// argument words, in the one layout the backend records them in.
///
/// - Vulkan writes an argument block: each buffer's and the scalar
///   results' address, then the words, from a 16-byte aligned start.
/// - A CUDA graph's memory does not follow its kernels' parameters: on an
///   RTX 3090, chains of 4,096 nodes of one word, of 1,040 bytes and of
///   4 KiB of parameters took 28 to 30 MiB alike
///   (`seismic-cuda` `graph_memory_probe`).
/// - Metal and the CPU record no arguments.
pub const fn argument_bytes(backend: BackendName, buffers: u64, words: u64) -> u64 {
    match backend {
        BackendName::Vulkan => (8 * (buffers + 1) + words).next_multiple_of(16),
        BackendName::Cuda | BackendName::Metal | BackendName::Cpu => 0,
    }
}

/// What a set of recorded graphs is made of, as far as their memory follows
/// it: the graphs, their launches, and every launch's argument bytes in the
/// backend's layout, with the largest one launch's. A set's census adds its members' (the largest
/// one's is theirs at most).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GraphCensus {
    pub graphs: u64,
    pub launches: u64,
    pub arguments: u64,
    pub largest: u64,
}

impl GraphCensus {
    /// One launch of `arguments` argument bytes, in no graph of its own: a
    /// graph's launches add up to its census, and the graph is
    /// [`Self::graph`].
    pub fn launch(arguments: u64) -> Self {
        Self {
            graphs: 0,
            launches: 1,
            arguments,
            largest: arguments,
        }
    }

    /// A graph of `launches`' census.
    pub fn graph(launches: Self) -> Option<Self> {
        Some(Self {
            graphs: launches.graphs.checked_add(1)?,
            ..launches
        })
    }

    /// The census of both sets.
    pub fn plus(self, other: Self) -> Option<Self> {
        Some(Self {
            graphs: self.graphs.checked_add(other.graphs)?,
            launches: self.launches.checked_add(other.launches)?,
            arguments: self.arguments.checked_add(other.arguments)?,
            largest: self.largest.max(other.largest),
        })
    }

    /// The census of what is left of this set without `other`, one of its
    /// parts (a part's largest launch stays this set's bound).
    pub fn minus(self, other: Self) -> Option<Self> {
        Some(Self {
            graphs: self.graphs.checked_sub(other.graphs)?,
            launches: self.launches.checked_sub(other.launches)?,
            arguments: self.arguments.checked_sub(other.arguments)?,
            largest: self.largest,
        })
    }

    /// The census of `count` sets of this one.
    pub fn times(self, count: u64) -> Option<Self> {
        Some(Self {
            graphs: self.graphs.checked_mul(count)?,
            launches: self.launches.checked_mul(count)?,
            arguments: self.arguments.checked_mul(count)?,
            largest: if count == 0 { 0 } else { self.largest },
        })
    }

    /// A census no less than either, term by term: what one of two sets
    /// holds at most.
    pub fn most(self, other: Self) -> Self {
        Self {
            graphs: self.graphs.max(other.graphs),
            launches: self.launches.max(other.launches),
            arguments: self.arguments.max(other.arguments),
            largest: self.largest.max(other.largest),
        }
    }

    /// Whether this set is no more than `bound` in every term: its memory
    /// is then no more than the bound's under any [`GraphCost`].
    pub fn within(self, bound: Self) -> bool {
        self.graphs <= bound.graphs
            && self.launches <= bound.launches
            && self.arguments <= bound.arguments
            && self.largest <= bound.largest
    }
}

/// What a family's graphs hold with launches in flight ([`Self::held`]):
/// the graphs it keeps live for each launch in flight (`pinned`), each a
/// decode launch's; the union of every launch's, each layout's for the most
/// runs of it any one launch makes; and what a chain of launches replays
/// beyond each launch's own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaunchGraphs {
    pub pinned: GraphCensus,
    pub union: GraphCensus,
    /// A chain of `k + 2` launches' graphs beyond each launch's own, at
    /// `k`: the further runs of the graphs it runs once a group, where one
    /// graph serves one run of its shape a submission. Empty where none.
    pub chained: Vec<GraphCensus>,
}

/// One graph a family's launches replay (a layout's, or a plan's sealed
/// alone): what recording it takes, the runs of it the family keeps live
/// for each launch in flight (the most any one of its decode launches
/// makes), and the most runs of it any one of its launches makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchedGraph {
    pub graph: GraphCensus,
    pub pinned_runs: u64,
    pub runs: u64,
}

impl LaunchGraphs {
    /// What a family of `graphs` holds, with `chained` its chains' further
    /// graphs: its pinned graphs, each graph for its pinned runs, all of
    /// them resident at once for each launch in flight (`record_decode`),
    /// so their sum; and its union, each graph for the most runs any one
    /// launch makes. The one computation of a family's census, from the
    /// graphs its plans are sealed into, for the planner's charge and the
    /// sealed family's alike. None on overflow.
    pub fn of(graphs: impl IntoIterator<Item = LaunchedGraph>, chained: Vec<GraphCensus>) -> Option<Self> {
        let (mut pinned, mut union) = (GraphCensus::default(), GraphCensus::default());
        for LaunchedGraph { graph, pinned_runs, runs } in graphs {
            assert!(pinned_runs <= runs, "a pinned graph is one the family's launches run");
            pinned = pinned.plus(graph.times(pinned_runs)?)?;
            union = union.plus(graph.times(runs)?)?;
        }
        Some(Self { pinned, union, chained })
    }

    /// What the family holds with `launches` launches in flight: the
    /// union, within which any one launch's graphs are whatever its class,
    /// so every submission fits; beside it each other launch's pinned
    /// graphs, so a decode launch in flight with others records no graph it
    /// once recorded; and what a chain of them replays beyond each launch's
    /// own (`chained`), the longest chain's past it. Two instances of one
    /// shape arise only from two runs of it in one submission, which the
    /// union and `chained` count, or from the pinned graphs: outside a chain
    /// a submission runs one launch (`ExecutorDomain::submit_chain`), each
    /// priming its own (`submit_head`), and a run that finds no idle
    /// instance of its shape drains the device before it chooses again
    /// (`Replays::record`'s prepare; on CUDA `cuda::record_run`), so a
    /// launch beside another needs no instance of its own. The pinned
    /// graphs are within this, as a graph's union runs are no fewer than
    /// its pinned runs, so no release ever takes them. A second prompt
    /// launch in flight still fits, releasing graphs only the first
    /// replays, which then record again. None without a launch, none on
    /// overflow.
    pub fn held(&self, launches: u64) -> Option<GraphCensus> {
        let chained = match launches {
            0 => return Some(GraphCensus::default()),
            1 => GraphCensus::default(),
            launches => usize::try_from(launches - 2)
                .ok()
                .and_then(|at| self.chained.get(at).or(self.chained.last()))
                .copied()
                .unwrap_or_default(),
        };
        self.pinned.times(launches - 1)?.plus(self.union)?.plus(chained)
    }

    /// Whether every term is within `bound`'s: what it holds at any count
    /// of launches, and what it pins, is then within what `bound` does.
    pub fn within(&self, bound: &Self) -> bool {
        self.pinned.within(bound.pinned)
            && self.union.within(bound.union)
            && self.chained.len() == bound.chained.len()
            && self.chained.iter().zip(&bound.chained).all(|(chained, bound)| chained.within(*bound))
    }
}

/// The memory a set of recorded graphs holds, by the storage it is in:
/// device memory beyond the device's allocations (driver memory), and
/// upload storage the backend allocates for them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphBytes {
    pub device: u64,
    pub upload: u64,
}

impl GraphBytes {
    /// What `self` holds beyond `held`, in each storage.
    pub fn beyond(self, held: Self) -> Self {
        Self {
            device: self.device.saturating_sub(held.device),
            upload: self.upload.saturating_sub(held.upload),
        }
    }
}

/// What one device's recorded graphs hold for their census. It serializes
/// as its kind and its terms, which is how it enters an identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphCost {
    /// Recording holds no device memory: the CPU records nothing, and a
    /// Metal schedule is host data.
    None,
    /// A driver that instantiates every graph into memory of its own,
    /// taken from the device in `granule`-byte steps and kept for later
    /// graphs once a graph is destroyed (CUDA). Measured on the device, each
    /// a slope: per graph executable, and per kernel node, whatever its
    /// parameters.
    Pooled { graph: u64, launch: u64, granule: u64 },
    /// Command memory the driver takes per launch recorded, measured on the
    /// device, and argument blocks the backend packs into upload chunks of
    /// `chunk` bytes, a graph's own (Vulkan): a block never spans chunks,
    /// so each chunk a graph fills holds more than `chunk` less its largest
    /// block.
    Chunked { launch: u64, chunk: u64 },
}

impl GraphCost {
    /// What graphs of `census` hold; none on overflow, or where a launch's
    /// arguments fill a whole chunk, which no recording admits.
    pub fn bytes(self, census: GraphCensus) -> Option<GraphBytes> {
        match self {
            Self::None => Some(GraphBytes::default()),
            Self::Pooled { graph, launch, granule } => {
                let device = census
                    .graphs
                    .checked_mul(graph)?
                    .checked_add(census.launches.checked_mul(launch)?)?
                    .checked_next_multiple_of(granule.max(1))?;
                Some(GraphBytes { device, upload: 0 })
            }
            Self::Chunked { launch, chunk } => {
                let fill = chunk.checked_sub(census.largest).filter(|fill| *fill > 0)?;
                let chunks = census.graphs.checked_add(census.arguments.div_ceil(fill))?;
                Some(GraphBytes {
                    device: census.launches.checked_mul(launch)?,
                    upload: chunks.checked_mul(chunk)?,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn census(graphs: u64, launches: u64, arguments: u64, largest: u64) -> GraphCensus {
        GraphCensus { graphs, launches, arguments, largest }
    }

    /// A census adds its parts' terms and takes the larger largest launch;
    /// removing a part leaves the rest, and a set is within itself.
    #[test]
    fn launches_in_flight_hold_the_union_and_each_further_ones_pins() {
        let (small, large) = (census(1, 3, 0, 0), census(1, 10, 0, 0));
        let graph = |graph, pinned_runs| LaunchedGraph { graph, pinned_runs, runs: 1 };
        // No pin (a head's): every launch in flight within the union.
        let unpinned = LaunchGraphs::of([graph(small, 0), graph(large, 0)], Vec::new()).unwrap();
        assert_eq!(unpinned.pinned, GraphCensus::default());
        assert_eq!(unpinned.held(1), Some(census(2, 13, 0, 0)));
        assert_eq!(unpinned.held(2), Some(census(2, 13, 0, 0)));
        assert_eq!(unpinned.held(3), Some(census(2, 13, 0, 0)));
        // A pinned decode graph (the readout's): the union and each further
        // launch's pins.
        let pinned = LaunchGraphs::of([graph(small, 1), graph(large, 0)], Vec::new()).unwrap();
        assert_eq!(pinned.pinned, small);
        assert_eq!(pinned.held(2), Some(census(3, 16, 0, 0)));
        assert_eq!(pinned.held(3), Some(census(4, 19, 0, 0)));
        assert!(unpinned.within(&pinned) && !pinned.within(&unpinned));
    }

    #[test]
    fn a_census_adds_its_parts() {
        let first = GraphCensus::graph(GraphCensus::launch(40)).unwrap();
        let second = GraphCensus::graph(GraphCensus::launch(16).plus(GraphCensus::launch(24)).unwrap()).unwrap();
        let both = first.plus(second).unwrap();
        assert_eq!(both, census(2, 3, 80, 40));
        assert_eq!(both.minus(first).unwrap(), census(1, 2, 40, 40));
        assert_eq!(second.times(3).unwrap(), census(3, 6, 120, 24));
        assert!(first.within(both) && second.within(both) && !both.within(first));
        assert_eq!(first.most(second), census(1, 2, 40, 40));
    }

    /// Pooled memory is each term's measured cost, rounded up to the
    /// driver's granule once for the whole set.
    #[test]
    fn pooled_memory_rounds_the_set_to_its_granule() {
        let cost = GraphCost::Pooled { graph: 1000, launch: 300, granule: 4096 };
        // 2 graphs, 10 launches: 2000 + 3000 = 5000, to 8192.
        let bytes = cost.bytes(census(2, 10, 160, 16)).unwrap();
        assert_eq!(bytes, GraphBytes { device: 8192, upload: 0 });
        assert_eq!(GraphCost::None.bytes(census(2, 10, 160, 16)), Some(GraphBytes::default()));
    }

    /// Chunked argument blocks take a chunk per graph more than their bytes
    /// over what a chunk holds past its largest block; command memory is
    /// per launch.
    #[test]
    fn chunked_arguments_take_whole_chunks() {
        let cost = GraphCost::Chunked { launch: 64, chunk: 1024 };
        // 3 graphs, 20 launches, 2000 argument bytes, largest 24: 3 + 2 chunks.
        let bytes = cost.bytes(census(3, 20, 2000, 24)).unwrap();
        assert_eq!(bytes, GraphBytes { device: 1280, upload: 5 * 1024 });
        assert_eq!(cost.bytes(census(1, 1, 1024, 1024)), None);
    }
}
