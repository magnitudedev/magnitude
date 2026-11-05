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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

/// What one device's recorded graphs hold for their census.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
