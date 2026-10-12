//! Seismic-owned decoder block graphs. The engine supplies semantic model
//! topology and external state; checked entry contracts derive every stage
//! edge and intermediate allocation.

use crate::operators::{self, MixerKind, PairedBlock, paired_block};
use crate::device_table::ResidentTable;
use crate::streamed_table::StreamedTable;
use crate::{
    ModelLoadPlan, ResidentTarget, ResourceLimits, StateResourcePlan, TargetBlockProgramSlot,
    TargetProgramPlan,
    native::{AttestedTarget, AttestedTargetBlock, OutputScales},
    operators::block::{
        self, BlockControlPorts, BlockStatePorts, FeedForwardEntries, MixerEntries, PerLayerParts,
    },
    operators::per_layer::graph::{
        CheckedPerLayerEntryEntries, PerLayerEntryEntries, PerLayerEntryPorts, per_layer_entry,
        per_layer_entry_class_slice,
    },
    programs::graph::GraphError,
    programs::graph::RowForm,
    programs::graph::draft::GraphDraft,
    programs::graph::tap::{TapEntry, TapPorts, TapPositions, Taps},
    programs::native_constants::{
        CheckedGraphFamilyResources, CheckedGraphResources, ConstantTensors, FamilyConstants,
        GraphConstant, distinct_storage_bytes,
    },
};
use magnitude_family_contracts::{
    Decoder, EmbeddingScale, SublayerIndex, TapPoint, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::embedding_rows;
use magnitude_state::LayerRef;
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, HostHeap, NativeGraphClassSlice,
    NativeGraphFamily, NativeGraphLayout, NativeGraphMetadata, NativeGraphPlan,
    NativeGraphStorageBytes, NativePort, WorkflowTensor,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// A block graph's class: (rows, history segments, request slots, the
/// history row tiles of its launch's rows it lists; 0 for a class that lists
/// none).
type BlockClass = (u64, u64, u64, u64);

/// Whether the prefill graphs of `slot`'s block come in classes that list
/// their launch's history row tiles: an attention block's, where the plan's
/// device of `backend` reads its heads' history decoded.
fn slot_lists(
    state: &StateResourcePlan,
    backend: BackendName,
    slot: TargetBlockProgramSlot,
) -> bool {
    match slot.mixer() {
        crate::MixerProgramSlot::Attention(binding) => {
            state.lists_history_tiles(backend, binding.shape.width)
        }
        _ => false,
    }
}

/// The classes of a block's graphs of one row class that list their
/// launch's history row tiles, beside the class that lists none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BlockListings {
    /// Their tile counts, ascending
    /// (`HistoryStorePlan::listed_tile_classes`); empty when the row class
    /// has none.
    tiles: Vec<u64>,
    /// Whether each is sealed for every history segment count, or for the
    /// most alone (`DecodedHistoryForm::lists_every_segment_count`).
    every_segment_count: bool,
}

impl BlockListings {
    /// The segment count of the listing class a launch over `segments`
    /// history segments takes, of at most `max_segments`.
    fn segments(&self, segments: u64, max_segments: u64) -> u64 {
        match self.every_segment_count {
            true => segments,
            false => max_segments,
        }
    }
}

/// The listing classes of block `index`'s graphs of `rows` rows: those of
/// an attention block's prefill rows, on a plan whose device of `backend`
/// has a form that reads its heads' history decoded
/// (`StateResourcePlan::lists_history_tiles`).
fn block_listings(
    state: &StateResourcePlan,
    backend: BackendName,
    slot: TargetBlockProgramSlot,
    index: usize,
    rows: u64,
) -> Result<BlockListings, String> {
    let form = match slot.mixer() {
        crate::MixerProgramSlot::Attention(binding) => {
            state.decoded_history_form(backend, binding.shape.width)
        }
        _ => None,
    };
    let Some(form) = form.filter(|_| !crate::operators::attention::graph::decodes(rows)) else {
        return Ok(BlockListings::default());
    };
    Ok(BlockListings {
        tiles: state
            .target_state()
            .layer_history(LayerRef::Target(index as u32))
            .ok_or("attention block has no history domain")?
            .store
            .listed_tile_classes(form)?,
        every_segment_count: form.lists_every_segment_count(),
    })
}

/// The classes a block's sealed graph distinguishes for one row class. An
/// attention block reads history segments and is independent of request
/// slots; each segment count has a class per `listed` tile count and one
/// that lists none. A recurrent block
/// binds exact per-slot bank tables (every slot count a launch of `rows`
/// rows can serve, up to the launch slot bound) and is independent of
/// segments. Everything else is per row.
fn block_classes(
    mixer: MixerKind,
    rows: u64,
    max_slots: u64,
    max_segments: u64,
    listings: &BlockListings,
) -> Vec<BlockClass> {
    match mixer {
        MixerKind::Attention => std::iter::successors(Some(1u64), |segments| {
            segments
                .checked_mul(2)
                .filter(|segments| *segments <= max_segments)
        })
        .flat_map(|segments| {
            let listed = listings
                .tiles
                .iter()
                .copied()
                .filter(move |_| listings.segments(segments, max_segments) == segments);
            std::iter::once(0)
                .chain(listed)
                .map(move |listed| (rows, segments, 1, listed))
        })
        .collect(),
        MixerKind::Recurrent => (1..=rows.min(max_slots))
            .map(|slots| (rows, 1, slots, 0))
            .collect(),
    }
}

/// The class of a block's graph serving a launch of `rows` rows over
/// `segments` history segments and `slots` requests, listing `listed` of the
/// launch's history row tiles.
fn block_class(mixer: MixerKind, rows: u64, segments: u64, slots: u64, listed: u64) -> BlockClass {
    match mixer {
        MixerKind::Attention => (rows, segments, 1, listed),
        MixerKind::Recurrent => (rows, 1, slots, 0),
    }
}

/// The admitted classes of one block for a `rows`-row launch, as one slice:
/// the row count, and every history segment count (attention) or request
/// slot count (recurrent) that launch can serve. A routed feed-forward's
/// grouped entries size their tile table from the row count alone.
fn block_class_slice(
    block: &PairedBlock,
    rows: u64,
    max_slots: u64,
    max_segments: u64,
    listed: &[u64],
) -> Result<NativeGraphClassSlice, String> {
    let mixer = block.mixer.kind();
    // The segment and slot counts are those of every listing class.
    let classes = block_classes(mixer, rows, max_slots, max_segments, &BlockListings::default());
    let slice = NativeGraphClassSlice::new()
        .dimension("M", [rows])
        .dimension("O", [rows]);
    let slice = match mixer {
        MixerKind::Attention if !listed.is_empty() => slice
            .dimension("R", classes.iter().map(|&(_, segments, ..)| segments))
            .dimension("HT", listed.iter().copied()),
        MixerKind::Attention => {
            slice.dimension("R", classes.iter().map(|&(_, segments, ..)| segments))
        }
        MixerKind::Recurrent => slice.dimension("B", classes.iter().map(|&(_, _, slots, _)| slots)),
    };
    block::feed_forward_class_slice(block, rows, slice)
}

/// The blocks' output scales as their holder knows them. They are values of
/// a block's sealed graphs, so blocks share plans only at equal scales.
#[derive(Clone, Copy)]
enum BlockOutputScales<'a> {
    /// As read from the artifact, per block.
    Read(&'a [OutputScales]),
    /// As the program plan fixes them: 1 for every sublayer that reads no
    /// layer scale (`reads_layer_scale`), and unknown for one that does,
    /// which the enumeration then takes as `UnreadScales` says.
    Planned(UnreadScales),
}

/// What an enumeration takes the layer scales it has not read to be. Blocks
/// of one shape share a plan exactly where their scales are equal, so the
/// two bound the plans a load seals from either side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnreadScales {
    /// No two equal: every block that reads one seals its own plans. No
    /// load seals more.
    Distinct,
    /// All equal: blocks of one shape seal one block's plans. No load seals
    /// fewer.
    Equal,
}

/// The block graphs of a target: the (block, class) plans a load seals and
/// those a block holds of an earlier block's.
struct BlockGraphEnumeration {
    /// The plans sealed, in sealing order: row classes, then blocks, then
    /// the block's classes of that row class.
    sealing: Vec<(usize, BlockClass)>,
    /// The plans an earlier block of the same shape and output scales
    /// seals, with that block.
    sharing: Vec<(usize, BlockClass, usize)>,
    /// Per block, its prefill rows' listing classes.
    listings: Vec<BlockListings>,
    launch_classes: std::collections::BTreeSet<BlockClass>,
    /// Whether every block's sharing was decided on known output scales:
    /// always under `BlockOutputScales::Read`, and under
    /// `BlockOutputScales::Planned` when no two blocks of one shape read a
    /// layer scale. Otherwise `sealing` bounds the plans a load seals, from
    /// the side its `UnreadScales` names.
    exact: bool,
}

/// Enumerate the block graphs of `plan` over `row_classes`: the one
/// enumeration of the sealer (`PreparedTargetGraphs::prepare`, with the
/// scales it read) and of the planner (`certify_target_family`, with the
/// planned ones), so the planned sealed graphs are the sealed ones.
#[allow(clippy::too_many_arguments)]
fn enumerate_block_graphs(
    backend: BackendName,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    mixers: &[MixerKind],
    shapes: &[BlockGraphShape],
    scales: BlockOutputScales<'_>,
    row_classes: &[u64],
    max_slots: u64,
    max_segments: u64,
) -> Result<BlockGraphEnumeration, String> {
    let blocks = plan.blocks();
    if blocks.len() != shapes.len() || blocks.len() != mixers.len() {
        return Err("planned target block count disagrees with geometry".into());
    }
    let unit = |index: usize| {
        !crate::native::output_tails(blocks[index])
            .into_iter()
            .any(crate::native::reads_layer_scale)
    };
    let equal_scales = |first: usize, index: usize| match scales {
        BlockOutputScales::Read(scales) => scales[first] == scales[index],
        BlockOutputScales::Planned(UnreadScales::Distinct) => unit(first) && unit(index),
        // Blocks of one shape read the same scales or none.
        BlockOutputScales::Planned(UnreadScales::Equal) => true,
    };
    let shared = (0..blocks.len())
        .map(|index| {
            (0..index)
                .find(|&first| shapes[first] == shapes[index] && equal_scales(first, index))
        })
        .collect::<Vec<_>>();
    let exact = match scales {
        BlockOutputScales::Read(_) => true,
        BlockOutputScales::Planned(_) => (0..blocks.len()).all(|index| {
            (0..index)
                .all(|first| shapes[first] != shapes[index] || (unit(first) && unit(index)))
        }),
    };
    let mut enumeration = BlockGraphEnumeration {
        sealing: Vec::new(),
        sharing: Vec::new(),
        listings: vec![BlockListings::default(); blocks.len()],
        launch_classes: std::collections::BTreeSet::new(),
        exact,
    };
    for &rows in row_classes {
        for (index, &slot) in blocks.iter().enumerate() {
            let listings = block_listings(state, backend, slot, index, rows)?;
            if !listings.tiles.is_empty() {
                enumeration.listings[index] = listings.clone();
            }
            for class in block_classes(mixers[index], rows, max_slots, max_segments, &listings) {
                enumeration.launch_classes.insert(class);
                match shared[index] {
                    Some(first) => enumeration.sharing.push((index, class, first)),
                    None => enumeration.sealing.push((index, class)),
                }
            }
        }
    }
    Ok(enumeration)
}

/// The mixer kind of each decoder block, which decides its graph classes.
fn block_mixers(geometry: &Decoder) -> Result<Vec<MixerKind>, String> {
    geometry
        .blocks
        .iter()
        .map(|block| {
            paired_block(block)
                .map(|paired| paired.mixer.kind())
                .map_err(|error| error.to_string())
        })
        .collect()
}

/// A count of sealed target graphs and the host heap that many hold. Each
/// term is exact for the graphs counted: a plan's heap follows from its
/// layout's topology (`NativeGraphLayout::plan_heap`), and so do the
/// lengths of the vectors its prepared graph owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SealedGraphsHeap {
    pub graphs: usize,
    /// The sealed plans (`NativeGraphPlan::host_heap`).
    pub plans: HostHeap,
    /// What stores the prepared block graphs (`block_graph_storage`): one
    /// shared allocation for each sealed block graph, and each block's
    /// vector of its classes.
    pub storage: HostHeap,
    /// The vectors the prepared block graphs own: each one's weight ports,
    /// constants and state ports (`PreparedTargetBlockGraph::owned_heap`).
    pub owned: HostHeap,
}

impl SealedGraphsHeap {
    /// The three terms together.
    pub fn heap(&self) -> Option<HostHeap> {
        self.plans.plus(self.storage)?.plus(self.owned)
    }
}

/// The target graphs a load seals and the host heap they hold, from the
/// plan alone (`certify_target_family`), bounded from both sides. Where
/// every block's sharing follows from the plan the two are one count, the
/// sealed one. Where blocks of one shape read layer scales, which the plan
/// does not know, they are apart: what holds a load to a budget reads
/// `at_most`, and what is held to the load's own holding reads `at_least`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedSealedGraphs {
    /// No load seals more: every block that reads a layer scale counted
    /// with plans of its own (`UnreadScales::Distinct`).
    pub at_most: SealedGraphsHeap,
    /// No load seals fewer: blocks of one shape counted with one block's
    /// plans (`UnreadScales::Equal`).
    pub at_least: SealedGraphsHeap,
    /// The host heap of the graphs' static bindings, which a load makes
    /// when it binds its resident weights (`BoundTargetGraphs::host_heap`):
    /// one binding for each entry graph and for every block's every class,
    /// shared plan or not, and the vectors that hold the blocks' bindings.
    /// Exact whether or not the count is: blocks that share a plan each
    /// bind it.
    pub bound: HostHeap,
    /// The host heap of the family's one copy of each distinct constant
    /// value its block graphs bind (`FamilyConstants::host_heap`). Exact
    /// whether or not the count is: blocks of one shape bind the same
    /// values.
    pub constants: HostHeap,
}

impl PlannedSealedGraphs {
    /// Whether the plan fixes the sealed count: `at_most` is `at_least`.
    pub fn exact(&self) -> bool {
        self.at_most == self.at_least
    }

    /// The most host bytes a load's sealed target graphs hold: their
    /// plans, the prepared graphs that hold them and the family's constant
    /// values.
    pub fn at_most_bytes(&self) -> Option<u64> {
        self.at_most.heap()?.bytes.checked_add(self.constants.bytes)
    }

    /// The least host bytes a load's sealed target graphs hold.
    pub fn at_least_bytes(&self) -> Option<u64> {
        self.at_least
            .heap()?
            .bytes
            .checked_add(self.constants.bytes)
    }
}

/// The host heap of the vectors a prepared block graph owns, from their
/// lengths: `weights` weight ports, `constants` constants and `state` state
/// ports. What the planner counts from a block's template and what
/// `PreparedTargetBlockGraph::prepare` leaves a graph holding.
fn owned_vectors_heap(weights: usize, constants: usize, state: usize) -> HostHeap {
    use std::mem::size_of;
    let mut heap = HostHeap::default();
    heap.hold(weights, size_of::<(WeightPort, NativePort)>());
    heap.hold(constants, size_of::<GraphConstant>());
    heap.hold(state, size_of::<NativePort>());
    heap
}

/// A prepared block graph in the shared allocation the blocks that seal
/// one plan hold it in: the graph and the allocation's two reference counts.
const SHARED_BLOCK_GRAPH_BYTES: usize =
    std::mem::size_of::<PreparedTargetBlockGraph>() + 2 * std::mem::size_of::<usize>();

/// The host heap that stores a target's prepared block graphs: the vector
/// of the blocks' vectors, each block's vector of its `classes[block]`
/// `(class, graph)` pairs, and one shared allocation for each of the
/// `sealed` graphs. What the planner counts and what
/// `PreparedTargetGraphs::prepare` allocates.
fn block_graph_storage(classes: &[usize], sealed: usize) -> Result<HostHeap, String> {
    use std::mem::size_of;
    classes
        .iter()
        .map(|count| {
            HostHeap::allocation(
                *count,
                size_of::<(BlockClass, Arc<PreparedTargetBlockGraph>)>(),
            )
        })
        .try_fold(
            HostHeap::allocation(
                classes.len(),
                size_of::<Vec<(BlockClass, Arc<PreparedTargetBlockGraph>)>>(),
            ),
            HostHeap::plus,
        )
        .and_then(|heap| {
            heap.plus(HostHeap::allocation(1, SHARED_BLOCK_GRAPH_BYTES).times(sealed as u64)?)
        })
        .ok_or_else(|| "prepared block graph storage overflows".to_owned())
}

/// The host heap of the vectors that hold a target's block bindings: one of
/// a vector for each block, and each block's of its `(class, binding)`
/// pairs, `bindings[block]` of them. What the planner counts and what
/// `BoundTargetGraphs` allocates.
fn block_binding_storage(bindings: &[usize]) -> Result<HostHeap, String> {
    use std::mem::size_of;
    bindings
        .iter()
        .map(|count| {
            HostHeap::allocation(*count, size_of::<(BlockClass, BoundNativeGraphPlan)>())
        })
        .try_fold(
            HostHeap::allocation(
                bindings.len(),
                size_of::<Vec<(BlockClass, BoundNativeGraphPlan)>>(),
            ),
            HostHeap::plus,
        )
        .ok_or_else(|| "target binding storage overflows".to_owned())
}

/// The planned graphs accumulated: each adds its layout's plan heap, and a
/// block graph the vectors its prepared graph owns.
#[derive(Default)]
struct PlannedHeap {
    graphs: usize,
    plans: HostHeap,
    owned: HostHeap,
}

impl PlannedHeap {
    fn seal(&mut self, layout: &NativeGraphLayout) -> Result<(), String> {
        self.graphs += 1;
        self.plans = self
            .plans
            .plus(layout.plan_heap())
            .ok_or("planned sealed graph heap overflows")?;
        Ok(())
    }

    fn seal_block(&mut self, certified: &CertifiedBlockGraph) -> Result<(), String> {
        self.seal(&certified.layout)?;
        self.owned = self
            .owned
            .plus(certified.owned)
            .ok_or("planned prepared graph heap overflows")?;
        Ok(())
    }
}

/// The sealed graphs' host heap against their layouts' planned heap.
#[derive(Default)]
struct SealedHeap {
    heap: HostHeap,
    /// The vectors the prepared block graphs own.
    owned: HostHeap,
    mismatched: usize,
}

impl SealedHeap {
    /// Count the vectors `block` owns, prepared into `certified`. Vectors
    /// of other lengths than its template's are a defect of the plan's
    /// accounting, as a plan of another heap is (`seal`).
    fn own(
        &mut self,
        block: &PreparedTargetBlockGraph,
        certified: &CertifiedBlockGraph,
        graph: impl Fn() -> String,
    ) -> Result<(), String> {
        let (held, planned) = (block.owned_heap(), certified.owned);
        self.owned = self
            .owned
            .plus(held)
            .ok_or("prepared graph heap overflows")?;
        if held != planned {
            self.mismatched += 1;
            if cfg!(debug_assertions) {
                return Err(format!(
                    "{} owns {} host bytes in {} allocations; its template planned {} in {}",
                    graph(),
                    held.bytes,
                    held.allocations,
                    planned.bytes,
                    planned.allocations
                ));
            }
        }
        Ok(())
    }

    /// Count `plan`, sealed into `layout`. A plan that holds another heap
    /// than its layout planned is a defect of the plan's accounting: an
    /// error in a debug build, and counted in the load's seal report in a
    /// release one.
    fn seal(
        &mut self,
        plan: &NativeGraphPlan,
        layout: &NativeGraphLayout,
        graph: impl Fn() -> String,
    ) -> Result<(), String> {
        let (held, planned) = (plan.host_heap(), layout.plan_heap());
        self.heap = self
            .heap
            .plus(held)
            .ok_or("sealed graph heap overflows")?;
        if held != planned {
            self.mismatched += 1;
            if cfg!(debug_assertions) {
                return Err(format!(
                    "{} holds {} host bytes in {} allocations; its layout planned {} in {}",
                    graph(),
                    held.bytes,
                    held.allocations,
                    planned.bytes,
                    planned.allocations
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct PreparedTargetGraphs {
    entries: BTreeMap<(u64, EntryTokens), PreparedTargetEntryGraph>,
    /// Per block, its graph for each class it distinguishes, ascending by
    /// class. Blocks that share a sealed plan share the prepared graph.
    blocks_by_class: Vec<Vec<(BlockClass, Arc<PreparedTargetBlockGraph>)>>,
    /// Per block, its prefill rows' listing classes (`block_listings`).
    listings: Vec<BlockListings>,
    /// The most history segments a launch has.
    max_segments: u64,
    /// Block mixers, which decide the class a launch selects per block.
    mixers: Vec<MixerKind>,
    classes: usize,
    family: NativeGraphFamily,
    max_output_bytes: u64,
    /// Decoder blocks, each one graph run per step.
    blocks: usize,
    /// The element and `[rows, taps · hidden]` extents of the draft input
    /// rows, when a separate draft taps the target.
    tap_buffer: Option<(Element, [u64; 2])>,
    /// Per block, the tap indices its graph writes.
    taps: Vec<BlockTaps>,
    /// The first block of the stateless tail (`stateless_tail`).
    stateless_tail: usize,
    /// The per-layer entry graph per row class, of a per-layer entry.
    per_layer_entries: BTreeMap<u64, PreparedPerLayerEntryGraph>,
    /// The `[rows, L·P]` F32 extents of the per-layer rows every per-layer
    /// input sublayer reads, over the largest row class.
    per_layer_rows: Option<[u64; 2]>,
    seal: SealReport,
}

/// The per-layer entry graph of one row class.
#[derive(Clone)]
pub(crate) struct PreparedPerLayerEntryGraph {
    pub plan: NativeGraphPlan,
    pub ports: PerLayerEntryPorts,
}

/// Sealing cost of a graph set: shape classes, graphs actually sealed after
/// sharing identical block plans, wall-clock seconds, and the host heap the
/// sealed plans hold against what the plan held them to before any was
/// sealed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SealReport {
    pub classes: usize,
    pub sealed_graphs: usize,
    pub seconds: f64,
    /// The graphs and host heap the load's plan counted
    /// (`PlannedSealedGraphs`).
    pub planned: PlannedSealedGraphs,
    /// The host heap the sealed plans hold (`NativeGraphPlan::host_heap`).
    pub heap: HostHeap,
    /// The host heap that stores the prepared block graphs, against
    /// `SealedGraphsHeap::storage`.
    pub storage: HostHeap,
    /// The host heap of the vectors the prepared block graphs own, against
    /// `SealedGraphsHeap::owned`.
    pub owned: HostHeap,
    /// The host heap of the family's constant values, against
    /// `PlannedSealedGraphs::constants`.
    pub constants: HostHeap,
    /// Sealed plans whose heap is not the one their layout planned, and
    /// prepared block graphs whose vectors are not of their template's
    /// lengths. Any is a defect of the plan's accounting; a debug build
    /// refuses the load.
    pub mismatched_graphs: usize,
}

impl SealReport {
    /// What the sealed graphs hold, in the plan's terms.
    pub fn held(&self) -> SealedGraphsHeap {
        SealedGraphsHeap {
            graphs: self.sealed_graphs,
            plans: self.heap,
            storage: self.storage,
            owned: self.owned,
        }
    }
}

/// The inputs that determine a block's sealed graph apart from its layer
/// index: its geometry, the resident element and shape of each weight role
/// within the block, and its recurrent state components. Blocks with equal
/// shapes share plans.
#[derive(Debug, PartialEq)]
struct BlockGraphShape {
    geometry: String,
    /// (sublayer, branch) of each role, its resident element, shape and
    /// accumulator-scale port extent.
    weights: Vec<((u32, Option<u32>), WeightKind, Element, Vec<u64>, u64)>,
    state: String,
    /// Where a separate draft taps the block.
    tapped: TapPositions,
}

/// A block's draft tap indices (column blocks of the draft input rows): at
/// its entry, before its feed-forward sublayer, and at its output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockTaps {
    pub entry: Option<u32>,
    pub middle: Option<u32>,
    pub output: Option<u32>,
}

impl BlockTaps {
    pub fn positions(self) -> TapPositions {
        TapPositions {
            entry: self.entry.is_some(),
            middle: self.middle.is_some(),
            output: self.output.is_some(),
        }
    }
}

/// The first block of the longest run of trailing blocks that publish no
/// state: attention blocks reading another layer's history (a Shared
/// domain, which they never append to) that write no draft taps. A row's
/// output from such a block reaches nothing but later blocks of the run and
/// the readout, so a step need run the run only over the rows its readout
/// consumes. The block count when the last block publishes state.
fn stateless_tail(state: &StateResourcePlan, mixers: &[MixerKind], taps: &[BlockTaps]) -> usize {
    trailing_run(mixers.len(), |index| {
        let layer = LayerRef::Target(index as u32);
        mixers[index] == MixerKind::Attention
            && taps[index] == BlockTaps::default()
            && state
                .target_state()
                .layer_history(layer)
                .is_some_and(|history| history.component.layer != layer)
    })
}

/// The first of the trailing blocks of `blocks` that all satisfy
/// `stateless`; `blocks` when the last does not.
fn trailing_run(blocks: usize, stateless: impl Fn(usize) -> bool) -> usize {
    (0..blocks)
        .rev()
        .take_while(|&index| stateless(index))
        .last()
        .unwrap_or(blocks)
}

/// The draft input row width (`taps · hidden`) when a separate draft taps
/// the target, and each block's taps. A tap entering sublayer 1 is the
/// residual entering the block's feed-forward; the exit tap is the last
/// block's output.
pub(crate) fn block_taps(
    plan: &TargetProgramPlan,
    geometry: &Decoder,
) -> Result<(Option<u64>, Vec<BlockTaps>), String> {
    let mut tapped = vec![BlockTaps::default(); geometry.blocks.len()];
    let Some(taps) = plan.taps() else {
        return Ok((None, tapped));
    };
    for (tap, point) in taps.points.iter().enumerate() {
        let tap = Some(u32::try_from(tap).map_err(|_| "tap index exceeds u32")?);
        let slot = match point {
            TapPoint::Sublayer(SublayerIndex { block, sublayer: 0 }) => {
                &mut tapped[*block as usize].entry
            }
            TapPoint::Sublayer(SublayerIndex { block, sublayer: 1 }) => {
                &mut tapped[*block as usize].middle
            }
            TapPoint::Exit => {
                &mut tapped
                    .last_mut()
                    .ok_or("a tapped target has no block")?
                    .output
            }
            TapPoint::Sublayer(index) => {
                return Err(format!("the target taps no residual entering {index:?}"));
            }
        };
        *slot = tap;
    }
    Ok((Some(taps.points.len() as u64 * geometry.hidden), tapped))
}

/// The weight scopes of a target block's mixer and feed-forward sublayers.
pub(crate) fn block_scopes(block: usize) -> Result<[WeightScope; 2], String> {
    let block = u32::try_from(block).map_err(|_| "target block index exceeds u32")?;
    Ok([0, 1].map(|sublayer| WeightScope::TargetSublayer(SublayerIndex { block, sublayer })))
}

/// A target sublayer (or branch) role moved to the same sublayer (or
/// branch) of `block`.
fn block_role(role: WeightRole, block: usize) -> Result<WeightRole, String> {
    let block = u32::try_from(block).map_err(|_| "target block index exceeds u32")?;
    let scope = match role.scope {
        WeightScope::TargetSublayer(SublayerIndex { sublayer, .. }) => {
            WeightScope::TargetSublayer(SublayerIndex { block, sublayer })
        }
        WeightScope::TargetBranch {
            sublayer: SublayerIndex { sublayer, .. },
            branch,
        } => WeightScope::TargetBranch {
            sublayer: SublayerIndex { block, sublayer },
            branch,
        },
        _ => {
            return Err(format!(
                "block graph weight {role:?} is not a target sublayer role"
            ));
        }
    };
    Ok(WeightRole {
        scope,
        kind: role.kind,
    })
}

fn block_graph_shapes(
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    tapped: &[BlockTaps],
) -> Result<Vec<BlockGraphShape>, String> {
    let components = &state.target_state().recurrent_components;
    geometry
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            let paired = paired_block(block).map_err(|error| error.to_string())?;
            let mut weights = load
                .weights()
                .filter_map(|weight| match weight.role.scope {
                    WeightScope::TargetSublayer(sublayer) if sublayer.block as usize == index => {
                        Some((
                            (sublayer.sublayer, None),
                            weight.role.kind,
                            weight.resident,
                            weight.shape.clone(),
                            weight.scale_extent(),
                        ))
                    }
                    WeightScope::TargetBranch { sublayer, branch }
                        if sublayer.block as usize == index =>
                    {
                        Some((
                            (sublayer.sublayer, Some(branch)),
                            weight.role.kind,
                            weight.resident,
                            weight.shape.clone(),
                            weight.scale_extent(),
                        ))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            weights.sort_by_key(|(sublayer, kind, ..)| (*sublayer, format!("{kind:?}")));
            let state = match paired.mixer.kind() {
                MixerKind::Recurrent => {
                    let first = operators::bank_component_index(&geometry.blocks, index)
                        .map_err(|error| error.to_string())?;
                    let count = paired.mixer.bank_components();
                    format!("{:?}", components.get(first..first + count))
                }
                // Blocks share a plan only within one history domain's rows
                // and slabs.
                MixerKind::Attention => {
                    let history = state
                        .target_state()
                        .layer_history(LayerRef::Target(
                            u32::try_from(index).map_err(|_| "block index exceeds u32")?,
                        ))
                        .ok_or("attention block has no history domain")?;
                    format!("{:?}", (history.store.rows, history.store.slab_rows))
                }
            };
            Ok(BlockGraphShape {
                geometry: paired.shape_key().map_err(|error| error.to_string())?,
                weights,
                state,
                tapped: tapped[index].positions(),
            })
        })
        .collect()
}

impl PreparedTargetGraphs {
    /// Bytes binding allocates: the distinct graph constants, the draft
    /// input rows of a tapped target, and the per-layer rows.
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        let taps = self
            .tap_buffer
            .map_or(Ok(0), |(element, extents)| {
                element.canonical_byte_len(&extents)
            })
            .map_err(|error| error.to_string())?;
        let per_layer_rows = per_layer_rows_bytes(self.per_layer_rows)?;
        distinct_storage_bytes(
            self.blocks_by_class
                .iter()
                .flat_map(|graphs| graphs.iter().flat_map(|(_, graph)| graph.constants.iter())),
        )?
        .checked_add(taps)
        .and_then(|bytes| bytes.checked_add(per_layer_rows))
        .ok_or_else(|| "target binding charge overflows".into())
    }

    pub(crate) fn prepare(
        device: &Device,
        handles: &AttestedTarget,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        workers: &seismic::Workers,
    ) -> Result<Self, String> {
        let mixers = block_mixers(geometry)?;
        // The plan was assessed for a device that forms tensor operations or
        // not; the opened device's own probe must say the same, or its
        // kernels and these graphs would be of different classes.
        if state.tensor_operations() != device.forms_tensor_operations() {
            return Err(format!(
                "the plan holds that the device {} Metal tensor operations, and the opened device {}",
                if state.tensor_operations() {
                    "forms"
                } else {
                    "does not form"
                },
                if device.forms_tensor_operations() {
                    "does"
                } else {
                    "does not"
                },
            ));
        }
        let certificate =
            certify_target_family(device.backend(), load, geometry, state, plan, limits)
                .map_err(|error| error.to_string())?;
        let row_classes = magnitude_batching::row_classes(limits.max_launch_rows);
        if row_classes.is_empty() {
            return Err(format!(
                "target batch row bound {} has no row class",
                limits.max_launch_rows
            ));
        }
        let largest_rows = *row_classes.last().expect("row classes are nonempty") as u64;
        let max_slots = u64::try_from(limits.max_launch_slots)
            .map_err(|_| "target request slot bound exceeds u64")?;
        let max_segments = u64::try_from(
            state
                .target_state()
                .span_limit()
                .checked_next_power_of_two()
                .ok_or("target segment class overflows")?,
        )
        .map_err(|_| "target segment class exceeds u64")?;
        // Blocks whose sealed graph would be identical share one plan per
        // class: the plan names weights by kind only, and each block binds
        // its own resident weights to it.
        let (tap_width, tapped) = block_taps(plan, geometry)?;
        let stateless_tail = stateless_tail(state, &mixers, &tapped);
        let shapes = block_graph_shapes(load, geometry, state, &tapped)?;
        // Output scales are values of the sealed graph.
        let scales = handles
            .blocks
            .iter()
            .map(|handle| handle.output_scales)
            .collect::<Vec<_>>();
        let row_classes = row_classes
            .into_iter()
            .map(|rows| rows as u64)
            .collect::<Vec<_>>();
        // The (block, class) plans to seal, and those an earlier block of
        // the same shape seals (with the block that does), as enumerated.
        let BlockGraphEnumeration {
            sealing,
            sharing,
            listings,
            launch_classes,
            exact: _,
        } = enumerate_block_graphs(
            device.backend(),
            state,
            plan,
            &mixers,
            &shapes,
            BlockOutputScales::Read(&scales),
            &row_classes,
            max_slots,
            max_segments,
        )?;
        let mut held = SealedHeap::default();
        let mut entries = BTreeMap::new();
        let per_layer = PerLayerEntryGraphSource::of(
            handles,
            load,
            geometry,
            plan,
            certificate.per_layer.as_ref(),
        )?;
        let mut per_layer_entries = BTreeMap::new();
        let mut plans = Vec::new();
        let mut max_output_bytes = 0u64;
        let trace = std::env::var_os("MAGNITUDE_V4_TRACE_GRAPH_SEAL").is_some();
        let began_all = Instant::now();
        let mut sealed_graphs = 0usize;
        for &rows in &row_classes {
            for source in [EntryTokens::Uploaded, EntryTokens::Selected] {
                let layout = &certificate.entries[&source];
                let entry = PreparedTargetEntryGraph::prepare(
                    device,
                    &handles.embedding,
                    load,
                    geometry,
                    rows,
                    source,
                    layout,
                )?;
                sealed_graphs += 1;
                held.seal(&entry.plan, layout, || {
                    format!("target entry graph {source:?} of row class {rows}")
                })?;
                max_output_bytes = max_output_bytes.max(entry.plan.output_bytes());
                plans.push(entry.plan.clone());
                entries.insert((rows, source), entry);
            }
            if let Some(per_layer) = &per_layer {
                let entry = per_layer.prepare(device, rows)?;
                sealed_graphs += 1;
                held.seal(&entry.plan, per_layer.layout, || {
                    format!("per-layer entry graph of row class {rows}")
                })?;
                max_output_bytes = max_output_bytes.max(entry.plan.output_bytes());
                plans.push(entry.plan.clone());
                per_layer_entries.insert(rows, entry);
            }
        }
        // A block's plan of one class is sealed from the model's plan and
        // the block's kernels alone: no plan reads another. They are sealed
        // on the load's worker threads, as many at once as it has, and
        // placed in the order enumerated. The constant values they bind are
        // the family's: each distinct one held once, whichever graphs bind
        // it.
        let constants = FamilyConstants::default();
        let seal_block = |&(index, class): &(usize, BlockClass)| {
            let (rows, segments, slots, listed) = class;
            let began = Instant::now();
            let positions = tapped[index].positions();
            let tap = match (positions.any(), tap_width, &handles.taps) {
                (false, ..) => None,
                (true, Some(width), Some(taps)) => Some(TapEntry {
                    entry: &taps.tap,
                    width,
                    positions,
                }),
                _ => return Err("a tapped block has no tap entry".to_owned()),
            };
            let block = PreparedTargetBlockGraph::prepare(
                device,
                &handles.blocks[index],
                tap,
                plan.blocks()[index].per_layer(),
                load,
                geometry,
                state,
                index,
                rows,
                segments,
                slots,
                listed,
                &certificate.blocks[index][&(RowForm::of(rows), listed > 0)].layout,
                &constants,
            )
            .map_err(|error| {
                format!(
                    "target graph row class {rows}, history segments {segments}, request slots {slots}, listed {listed}, block {index}: {error}"
                )
            })?;
            if trace {
                eprintln!(
                    "target graph sealed rows={rows} segments={segments} slots={slots} block={index} elapsed_ms={:.3}",
                    began.elapsed().as_secs_f64() * 1000.0
                );
            }
            Ok(block)
        };
        let sealed = workers.each(&sealing, seal_block);
        // Each block's graphs go into a vector of exactly its classes, in
        // the order enumerated, which ascends by class.
        let mut class_counts = vec![0usize; handles.blocks.len()];
        for index in sealing
            .iter()
            .map(|&(index, _)| index)
            .chain(sharing.iter().map(|&(index, ..)| index))
        {
            class_counts[index] += 1;
        }
        let mut blocks_by_class = class_counts
            .iter()
            .map(|&classes| Vec::with_capacity(classes))
            .collect::<Vec<Vec<(BlockClass, Arc<PreparedTargetBlockGraph>)>>>();
        for (&(index, class), block) in sealing.iter().zip(sealed) {
            let block = block?;
            sealed_graphs += 1;
            let (rows, _, _, listed) = class;
            let certified = &certificate.blocks[index][&(RowForm::of(rows), listed > 0)];
            let graph = || format!("target graph class {class:?} of block {index}");
            held.seal(&block.plan, &certified.layout, graph)?;
            held.own(&block, certified, graph)?;
            max_output_bytes = max_output_bytes.max(block.plan.output_bytes());
            plans.push(block.plan.clone());
            blocks_by_class[index].push((class, Arc::new(block)));
        }
        // A block whose sealed graph is another's holds that block's
        // prepared graph: the one plan, ports and constants.
        for (index, class, first) in sharing {
            let graphs = &blocks_by_class[first];
            let at = graphs
                .binary_search_by(|(sealed, _)| sealed.cmp(&class))
                .map_err(|_| format!("block {first} sealed no graph of class {class:?}"))?;
            let graph = graphs[at].1.clone();
            blocks_by_class[index].push((class, graph));
        }
        if blocks_by_class
            .iter()
            .any(|graphs| !graphs.is_sorted_by(|(first, _), (second, _)| first < second))
        {
            return Err("a block's graph classes were not enumerated in ascending order".into());
        }
        let storage = block_graph_storage(&class_counts, sealing.len())?;
        let held_storage = blocks_by_class
            .iter()
            .try_fold(
                HostHeap::allocation(
                    blocks_by_class.capacity(),
                    std::mem::size_of::<Vec<(BlockClass, Arc<PreparedTargetBlockGraph>)>>(),
                ),
                |heap, graphs| {
                    heap.plus(HostHeap::allocation(
                        graphs.capacity(),
                        std::mem::size_of::<(BlockClass, Arc<PreparedTargetBlockGraph>)>(),
                    ))
                },
            )
            .and_then(|heap| {
                heap.plus(
                    HostHeap::allocation(1, SHARED_BLOCK_GRAPH_BYTES)
                        .times(sealing.len() as u64)?,
                )
            })
            .ok_or("prepared block graph storage overflows")?;
        let held_constants = constants.host_heap()?;
        let seal_seconds = began_all.elapsed().as_secs_f64();
        // The plan counted these graphs before any was sealed, from both
        // sides: one count where it could tell which blocks share. What
        // was sealed and is held lies between its bounds, term by term,
        // and the constant values are the ones it counted.
        let planned = certificate.sealed;
        let held_graphs = SealedGraphsHeap {
            graphs: sealed_graphs,
            plans: held.heap,
            storage: held_storage,
            owned: held.owned,
        };
        let between = |term: fn(&SealedGraphsHeap) -> (u64, u64)| {
            let ((least_bytes, least), (bytes, allocations), (most_bytes, most)) = (
                term(&planned.at_least),
                term(&held_graphs),
                term(&planned.at_most),
            );
            least_bytes <= bytes && bytes <= most_bytes && least <= allocations && allocations <= most
        };
        fn heap(heap: HostHeap) -> (u64, u64) {
            (heap.bytes, heap.allocations)
        }
        if cfg!(debug_assertions)
            && !(held_storage == storage
                && held_constants == planned.constants
                && between(|graphs| (graphs.graphs as u64, graphs.graphs as u64))
                && between(|graphs| heap(graphs.plans))
                && between(|graphs| heap(graphs.storage))
                && between(|graphs| heap(graphs.owned)))
        {
            return Err(format!(
                "the sealed target graphs hold {held_graphs:?} with constants {held_constants:?}; \
                 the plan counted {planned:?}"
            ));
        }
        let family = NativeGraphFamily::new(&plans).map_err(|error| error.to_string())?;
        let seal = SealReport {
            classes: launch_classes.len(),
            sealed_graphs,
            seconds: seal_seconds,
            planned,
            heap: held.heap,
            storage: held_storage,
            owned: held.owned,
            constants: held_constants,
            mismatched_graphs: held.mismatched,
        };
        Ok(Self {
            entries,
            blocks_by_class,
            listings,
            max_segments,
            mixers,
            classes: launch_classes.len(),
            family,
            max_output_bytes,
            blocks: handles.blocks.len(),
            tap_buffer: tap_width.map(|width| (activation(geometry), [largest_rows, width])),
            taps: tapped,
            stateless_tail,
            per_layer_entries,
            per_layer_rows: per_layer_rows(plan, largest_rows),
            seal,
        })
    }

    /// The tile counts of block `index`'s listing classes for a launch of
    /// `rows` rows, ascending; empty when that row class has none.
    pub(crate) fn listed_tiles(&self, index: usize, rows: u64) -> &[u64] {
        match self.listings.get(index) {
            Some(listings) if !crate::operators::attention::graph::decodes(rows) => {
                &listings.tiles
            }
            _ => &[],
        }
    }

    /// The segment count of block `index`'s listing class for a launch over
    /// `segments` history segments: the launch's own, or the most where the
    /// block's listing classes are sealed for that alone.
    pub(crate) fn listed_segments(&self, index: usize, segments: u64) -> u64 {
        self.listings[index].segments(segments, self.max_segments)
    }

    /// The tap indices block `index` writes.
    pub fn block_taps(&self, index: usize) -> BlockTaps {
        self.taps[index]
    }

    /// The first block of the stateless tail: from it on, no block publishes
    /// state, so only the rows a step's readout consumes need its blocks
    /// (`stateless_tail`). The block count when there is no such tail.
    pub(crate) fn stateless_tail(&self) -> usize {
        self.stateless_tail
    }

    /// How many class graphs sealing formed and how long it took.
    pub fn seal_report(&self) -> SealReport {
        self.seal
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub fn output_bytes(&self) -> u64 {
        self.max_output_bytes
    }

    /// Graph runs one step queues on a workspace slot before any completes:
    /// the embedding entry, the per-layer entry of a per-layer model, and
    /// every block.
    pub fn runs_per_step(&self) -> usize {
        1 + usize::from(self.per_layer_rows.is_some()) + self.blocks
    }
}

/// The graph runs one target step queues (`PreparedTargetGraphs::
/// runs_per_step`), from the decoder: the embedding entry, the per-layer
/// entry of a per-layer decoder, and every block.
pub(crate) fn target_runs_per_step(decoder: &Decoder) -> usize {
    1 + usize::from(decoder.entry.per_layer.is_some()) + decoder.blocks.len()
}

impl PreparedTargetGraphs {
    pub fn class_count(&self) -> usize {
        self.classes
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    /// Bind every graph to the resident weights. The bound graphs share
    /// these prepared graphs; what they add on the host is one binding a
    /// graph and the vectors that hold them, which the load's plan counted
    /// before any graph was sealed (`PlannedSealedGraphs::bound`). Each
    /// block's bindings are built from that block alone, into a vector of
    /// its own.
    pub(crate) fn bind_weights(
        self: &Arc<Self>,
        resident: &ResidentTarget,
    ) -> Result<BoundTargetGraphs, String> {
        let embedding = resident.embedding.table().clone();
        let direct =
            seismic::Tensor::from_host(&resident.embedding.tensor().device(), Element::i32(), &[0], &[])
                .map_err(|error| error.to_string())?;
        let mut entry_bound = BTreeMap::new();
        for (key, entry) in &self.entries {
            let directory = match (key.1, embedding.directory()) {
                (EntryTokens::Selected, Some(directory)) => directory,
                _ => &direct,
            };
            let fixed = [
                (&entry.table, resident.embedding.tensor()),
                (&entry.directory, directory),
            ];
            entry_bound.insert(
                *key,
                entry
                    .plan
                    .bind_static(&fixed)
                    .map_err(|error| format!("target embedding graph class {key:?}: {error}"))?,
            );
        }
        let mut constants = ConstantTensors::new(resident.embedding.tensor().device());
        let mut bound = Vec::with_capacity(self.blocks_by_class.len());
        if resident.blocks != self.blocks_by_class.len() {
            return Err("resident target blocks disagree with the prepared graphs".into());
        }
        for (index, graphs) in self.blocks_by_class.iter().enumerate() {
            // The block's classes ascend in its map, so do its bindings.
            let mut block_bound = Vec::with_capacity(graphs.len());
            for (class, graph) in graphs {
                let constant_tensors = graph
                    .constants
                    .iter()
                    .map(|constant| Ok((constant.port(), constants.tensor(constant)?)))
                    .collect::<Result<Vec<_>, String>>()?;
                let fixed = graph
                    .weights
                    .iter()
                    .map(|(weight, port)| {
                        // A graph shared by equal-shape blocks names the
                        // roles of the block it was prepared for; each block
                        // binds its own.
                        let role = block_role(weight.role, index)?;
                        Ok((port, weight.part.of(resident.sublayers.get(role)?)?))
                    })
                    .chain(
                        constant_tensors
                            .iter()
                            .map(|(port, tensor)| Ok((*port, tensor))),
                    )
                    .collect::<Result<Vec<_>, String>>()?;
                block_bound.push((
                    *class,
                    graph.plan.bind_static(&fixed).map_err(|error| {
                        format!(
                            "target graph static binding class {class:?} block {index}: {error}"
                        )
                    })?,
                ));
            }
            bound.push(block_bound);
        }
        let taps = self
            .tap_buffer
            .map(|(element, extents)| {
                seismic::Tensor::zeros(&resident.embedding.tensor().device(), element, &extents)
                    .map_err(|error| format!("draft input rows allocation failed: {error}"))
            })
            .transpose()?;
        let per_layer = match (&resident.per_layer, self.per_layer_rows) {
            (None, None) => None,
            (Some(weights), Some(extents)) => Some(BoundPerLayerEntry {
                bound: self
                    .per_layer_entries
                    .iter()
                    .map(|(rows, graph)| {
                        let absent_scale = seismic::Tensor::from_host(
                            &weights.projection.tensor().device(),
                            Element::f32(),
                            &[0],
                            &[],
                        )
                        .map_err(|error| error.to_string())?;
                        let mut fixed = vec![
                            (&graph.ports.projection, weights.projection.tensor()),
                            (&graph.ports.norm, weights.norm.tensor()),
                            (&graph.ports.absent_scale, &absent_scale),
                        ];
                        if let Some(port) = &graph.ports.projection_bound {
                            let bound = weights
                                .projection
                                .bound()
                                .ok_or("the per-layer projection has no packing bound")?;
                            fixed.push((port, bound));
                        }
                        graph
                            .plan
                            .bind_static(&fixed)
                            .map(|bound| (*rows, bound))
                            .map_err(|error| format!("per-layer entry graph rows {rows}: {error}"))
                    })
                    .collect::<Result<_, String>>()?,
                // The rows a single-row step leaves unwritten hold tokens'
                // rows.
                token_rows: TokenRows::new(weights.table.rows(), extents[0] as usize - 1),
                table: StreamedTable::new(weights.table.clone()),
                rows: seismic::Tensor::zeros(
                    &resident.embedding.tensor().device(),
                    Element::f32(),
                    &extents,
                )
                .map_err(|error| format!("per-layer rows allocation failed: {error}"))?,
            }),
            _ => return Err("resident per-layer weights disagree with the prepared graphs".into()),
        };
        let graphs = BoundTargetGraphs {
            prepared: self.clone(),
            entry_bound,
            bound,
            constants: constants.into_tensors(),
            taps,
            per_layer,
            embedding,
        };
        // A binding that holds another heap than the plan counted is a
        // defect of the plan's accounting: an error in a debug build.
        let (held, planned) = (graphs.host_heap()?, self.seal.planned.bound);
        eprintln!(
            "magnitude-engine: bound the target graphs: {} host bytes in {} allocations ({} in {} planned)",
            held.bytes, held.allocations, planned.bytes, planned.allocations
        );
        if cfg!(debug_assertions) && held != planned {
            return Err(format!(
                "the bound target graphs hold {} host bytes in {} allocations; the plan counted {} in {}",
                held.bytes, held.allocations, planned.bytes, planned.allocations
            ));
        }
        Ok(graphs)
    }
}

/// The per-layer entry graph's inputs apart from its row class.
struct PerLayerEntryGraphSource<'a> {
    kernels: &'a crate::native::PerLayerEntryKernels,
    binding: crate::PerLayerEntryBinding,
    entry: &'a magnitude_family_contracts::PerLayerEntry,
    projection: &'a crate::WeightPlan,
    norm: &'a crate::WeightPlan,
    layout: &'a NativeGraphLayout,
}

impl<'a> PerLayerEntryGraphSource<'a> {
    fn of(
        handles: &'a AttestedTarget,
        load: &'a ModelLoadPlan,
        geometry: &'a Decoder,
        plan: &TargetProgramPlan,
        layout: Option<&'a NativeGraphLayout>,
    ) -> Result<Option<Self>, String> {
        match (
            &handles.per_layer,
            plan.per_layer(),
            &geometry.entry.per_layer,
            layout,
        ) {
            (None, None, None, None) => Ok(None),
            (Some(kernels), Some(binding), Some(entry), Some(layout)) => Ok(Some(Self {
                kernels,
                binding,
                entry,
                projection: target_weight(load, WeightKind::PerLayerModelProjection)?,
                norm: target_weight(load, WeightKind::PerLayerProjectionNorm)?,
                layout,
            })),
            _ => Err("the per-layer entry's kernels, binding and geometry disagree".into()),
        }
    }

    fn prepare(&self, device: &Device, rows: u64) -> Result<PreparedPerLayerEntryGraph, String> {
        let mut graph = device.native_graph_with_layout(self.layout);
        let ports = per_layer_entry(
            &mut graph,
            PerLayerEntryEntries::from(self.kernels),
            self.projection,
            self.norm,
            self.binding,
            self.entry,
            rows,
        )
        .map_err(|error| format!("per-layer entry graph rows {rows}: {error}"))?;
        let plan = GraphDraft::seal(graph).map_err(|error| error.to_string())?;
        // It writes the bound per-layer rows and exports nothing.
        if plan.output_bytes() != 0 {
            return Err(format!("per-layer entry graph rows {rows} exports outputs"));
        }
        Ok(PreparedPerLayerEntryGraph { plan, ports })
    }
}

/// The per-layer rows' extents over `rows` batch rows, of a per-layer entry.
fn per_layer_rows(plan: &TargetProgramPlan, rows: u64) -> Option<[u64; 2]> {
    plan.per_layer()
        .map(|binding| [rows, binding.layers * binding.width])
}

fn per_layer_rows_bytes(extents: Option<[u64; 2]>) -> Result<u64, String> {
    extents.map_or(Ok(0), |extents| {
        Element::f32()
            .canonical_byte_len(&extents)
            .map_err(|error| error.to_string())
    })
}

pub(crate) struct BoundTargetGraphs {
    /// The prepared graphs these bind: the load's one copy, shared with
    /// its attested programs. They are immutable once sealed.
    pub prepared: Arc<PreparedTargetGraphs>,
    pub entry_bound: BTreeMap<(u64, EntryTokens), BoundNativeGraphPlan>,
    /// Per block, its bound graph for each class it distinguishes,
    /// ascending by class. A vector of exactly the block's classes: what
    /// the plan counted (`block_binding_storage`).
    bound: Vec<Vec<(BlockClass, BoundNativeGraphPlan)>>,
    constants: Vec<seismic::Tensor>,
    /// The draft input rows of a tapped target: tapped blocks write their
    /// column blocks of a step's leading rows, and the readout fuses them.
    pub taps: Option<seismic::Tensor>,
    pub per_layer: Option<BoundPerLayerEntry>,
    /// The token embedding as the entries gather it.
    pub embedding: ResidentTable,
}

impl BoundTargetGraphs {
    /// The host heap the bindings hold beyond the prepared graphs they
    /// share: each binding's own (`BoundNativeGraphPlan::host_heap`) and the
    /// vectors that hold the blocks' bindings, at their capacities. The
    /// nodes of the two small maps of entry bindings are the allocator's
    /// and are not in it.
    fn host_heap(&self) -> Result<HostHeap, String> {
        let entries = self.entry_bound.values().chain(
            self.per_layer
                .iter()
                .flat_map(|per_layer| per_layer.bound.values()),
        );
        let blocks = self.bound.iter().flatten().map(|(_, bound)| bound);
        let bindings = entries
            .chain(blocks)
            .try_fold(HostHeap::default(), |heap, bound| {
                heap.plus(bound.host_heap())
            })
            .ok_or("bound graph heap overflows")?;
        // Each block's vector was made with exactly its classes' capacity,
        // and the outer one with the blocks'.
        let held = self.bound.iter().map(Vec::capacity).collect::<Vec<_>>();
        if self.bound.capacity() != held.len() {
            return Err("the blocks' bindings are held in more than their vector".into());
        }
        bindings
            .plus(block_binding_storage(&held)?)
            .ok_or_else(|| "bound graph heap overflows".to_owned())
    }
}

/// The bound per-layer entry of a per-layer model.
pub(crate) struct BoundPerLayerEntry {
    /// The graph per row class, bound to its weights.
    bound: BTreeMap<u64, BoundNativeGraphPlan>,
    /// The table whose rows each step gathers, streamed.
    pub table: StreamedTable,
    /// The per-layer rows: the entry writes a step's leading rows, and every
    /// per-layer input sublayer reads them.
    pub rows: seismic::Tensor,
    /// The per-layer rows the entry computed for single tokens, kept in the
    /// rows a single-row step leaves unwritten.
    pub token_rows: TokenRows,
}

/// Where a single-row step of a text token finds its per-layer row. A text
/// token's per-layer row is its own alone, so the row a step computed for a
/// token stays where the step wrote it, and a later single-row step of the
/// token reads it there instead of running the entry. The rows held are the
/// per-layer rows after the first, the last row the first slot; a step of
/// more rows writes over the ones its rows cover.
pub(crate) struct TokenRows(std::sync::Mutex<crate::slot_directory::SlotDirectory>);

/// Where a single-row step of a token finds its per-layer row.
pub(crate) enum TokenRow {
    /// An earlier step wrote it into this slot.
    Held(usize),
    /// No slot holds it, and this one is free: the step that writes it
    /// there holds it (`TokenRows::hold`).
    Free(usize),
    /// No slot holds it, and none is free.
    None,
}

impl TokenRows {
    /// Rows of `tokens` tokens, in `slots` slots.
    fn new(tokens: u64, slots: usize) -> Self {
        Self(std::sync::Mutex::new(
            crate::slot_directory::SlotDirectory::new(tokens, slots as u64, 1),
        ))
    }

    /// Where a single-row step of `token` finds its per-layer row.
    pub(crate) fn find(&self, token: u64) -> Result<TokenRow, String> {
        let mut slots = self.0.lock().expect("token rows lock");
        if !slots.has_row(token) {
            return Err(format!("token {token} has no per-layer row"));
        }
        Ok(match slots.use_slot(token) {
            Some(slot) => TokenRow::Held(slot as usize),
            None => slots
                .free(|_| false)
                .map_or(TokenRow::None, |(slot, _)| TokenRow::Free(slot)),
        })
    }

    /// The step that writes `token`'s row into the free slot `slot` is
    /// queued: steps after it find the row there.
    pub(crate) fn hold(&self, token: u64, slot: usize) {
        self.0.lock().expect("token rows lock").hold(token, slot, true);
    }

    /// A step writes other rows over the slots `slots`.
    pub(crate) fn overwrite(&self, slots: std::ops::Range<usize>) {
        let mut held = self.0.lock().expect("token rows lock");
        for slot in slots {
            held.release(slot);
        }
    }
}

#[cfg(test)]
mod token_rows_tests {
    use super::{TokenRow, TokenRows};

    #[test]
    fn a_token_row_is_held_from_the_step_that_writes_it_until_overwritten() {
        assert!(matches!(TokenRows::new(8, 0).find(3).unwrap(), TokenRow::None));
        assert!(TokenRows::new(8, 2).find(8).is_err());

        let rows = TokenRows::new(8, 2);
        let TokenRow::Free(first) = rows.find(3).unwrap() else {
            panic!("a token no step wrote has a free slot")
        };
        // The slot is the token's only once the writing step is queued.
        assert!(matches!(rows.find(3).unwrap(), TokenRow::Free(_)));
        rows.hold(3, first);
        assert!(matches!(rows.find(3).unwrap(), TokenRow::Held(slot) if slot == first));
        let TokenRow::Free(second) = rows.find(4).unwrap() else {
            panic!("the other slot is free")
        };
        assert_ne!(second, first);
        rows.hold(4, second);
        assert!(matches!(rows.find(4).unwrap(), TokenRow::Held(slot) if slot == second));
        // A third token takes a slot from one of the two.
        let TokenRow::Free(third) = rows.find(5).unwrap() else {
            panic!("a full directory frees a slot")
        };
        rows.hold(5, third);
        assert!(matches!(rows.find(5).unwrap(), TokenRow::Held(slot) if slot == third));
        rows.overwrite(0..2);
        assert!(matches!(rows.find(5).unwrap(), TokenRow::Free(_)));
    }
}

impl BoundTargetGraphs {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.constants
            .iter()
            .chain(&self.taps)
            .chain(self.per_layer.as_ref().map(|per_layer| &per_layer.rows))
            .try_fold(0u64, |bytes, tensor| {
                bytes
                    .checked_add(tensor.storage_bytes())
                    .ok_or("target graph constant charge overflows")
            })
    }

    pub(crate) fn entry(
        &self,
        rows: u64,
        source: EntryTokens,
    ) -> Result<(&PreparedTargetEntryGraph, &BoundNativeGraphPlan), String> {
        let key = (rows, source);
        let graph = self
            .prepared
            .entries
            .get(&key)
            .ok_or_else(|| format!("target embedding class {key:?} was not sealed"))?;
        let bound = self
            .entry_bound
            .get(&key)
            .ok_or_else(|| format!("target embedding class {key:?} was not bound"))?;
        Ok((graph, bound))
    }

    /// The per-layer entry graph of row class `rows`, of a per-layer model.
    #[allow(clippy::type_complexity)]
    pub(crate) fn per_layer_entry(
        &self,
        rows: u64,
    ) -> Option<
        Result<
            (
                &PreparedPerLayerEntryGraph,
                &BoundNativeGraphPlan,
                &BoundPerLayerEntry,
            ),
            String,
        >,
    > {
        let per_layer = self.per_layer.as_ref()?;
        Some(
            self.prepared
                .per_layer_entries
                .get(&rows)
                .zip(per_layer.bound.get(&rows))
                .map(|(graph, bound)| (graph, bound, per_layer))
                .ok_or_else(|| format!("per-layer entry class {rows} was not sealed and bound")),
        )
    }

    pub(crate) fn block(
        &self,
        rows: u64,
        segments: u64,
        slots: u64,
        listed: u64,
        index: usize,
    ) -> Result<(&PreparedTargetBlockGraph, &BoundNativeGraphPlan), String> {
        let mixer = *self
            .prepared
            .mixers
            .get(index)
            .ok_or_else(|| format!("target block {index} does not exist"))?;
        let key = block_class(mixer, rows, segments, slots, listed);
        let graphs = &self.prepared.blocks_by_class[index];
        let graph = graphs
            .binary_search_by(|(class, _)| class.cmp(&key))
            .map(|at| graphs[at].1.as_ref())
            .map_err(|_| format!("target graph class {key:?} block {index} was not sealed"))?;
        let bound = &self.bound[index];
        let bound = bound
            .binary_search_by(|(class, _)| class.cmp(&key))
            .map(|at| &bound[at].1)
            .map_err(|_| format!("target graph class {key:?} block {index} was not bound"))?;
        Ok((graph, bound))
    }
}

/// How an entry graph receives its `[rows, 2]` (token, status) input rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum EntryTokens {
    /// Written by the host into the step's upload region.
    Uploaded,
    /// Bound to the previous step's selection tensor on the device.
    Selected,
}

#[derive(Clone)]
pub(crate) struct PreparedTargetEntryGraph {
    pub plan: NativeGraphPlan,
    pub table: NativePort,
    /// The directory the embedding addresses the table through: a streamed
    /// table's for selected tokens, else empty (`RowDirectory`).
    pub directory: NativePort,
    pub tokens: NativePort,
    pub hidden: WorkflowTensor,
}

impl PreparedTargetEntryGraph {
    fn prepare(
        device: &Device,
        embedding: &seismic::NativeKernel<embedding_rows::Entry>,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        rows: u64,
        source: EntryTokens,
        layout: &NativeGraphLayout,
    ) -> Result<Self, String> {
        let draft = entry_graph_topology(
            device.native_graph_with_layout(layout),
            embedding,
            embedding_weight(load)?,
            geometry,
            rows,
            source,
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            plan: GraphDraft::seal(draft.graph).map_err(|error| error.to_string())?,
            table: draft.table,
            directory: draft.directory,
            tokens: draft.tokens,
            hidden: draft.hidden,
        })
    }
}

fn embedding_weight(load: &ModelLoadPlan) -> Result<&crate::WeightPlan, String> {
    target_weight(load, WeightKind::Embedding)
}

fn target_weight(load: &ModelLoadPlan, kind: WeightKind) -> Result<&crate::WeightPlan, String> {
    let role = WeightRole {
        scope: WeightScope::Target,
        kind,
    };
    load.weights()
        .find(|weight| weight.role == role)
        .ok_or_else(|| format!("target weight {kind:?} is absent"))
}

/// An entry graph before it is sealed, and its external ports.
struct EntryGraphDraft<G> {
    graph: G,
    table: NativePort,
    directory: NativePort,
    tokens: NativePort,
    hidden: WorkflowTensor,
}

fn entry_graph_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    embedding: G::Binding<'a, embedding_rows::Entry>,
    weight: &crate::WeightPlan,
    geometry: &Decoder,
    rows: u64,
    source: EntryTokens,
) -> Result<EntryGraphDraft<G>, GraphError> {
    let table = graph.port(weight.resident, &weight.shape)?;
    // Uploaded tokens are the host's: of a streamed table it writes their
    // slots. Selected tokens are the device's, which finds their slots in
    // the directory.
    let entries = match source {
        EntryTokens::Uploaded => 0,
        EntryTokens::Selected => weight.directory_entries(),
    };
    let dimensions = [
        ("M", rows),
        ("V", weight.shape[0]),
        ("D", geometry.hidden),
        ("R", entries),
    ];
    let tokens = match source {
        EntryTokens::Uploaded => graph.input_for(embedding, "tokens", &dimensions)?,
        EntryTokens::Selected => {
            graph.port_with_class_extent(Element::i32(), &[rows, 2], 0, "M")?
        }
    };
    let directory = graph.port(Element::i32(), &[entries])?;
    let (scale, normalize, epsilon) = embedding_transform(geometry);
    let result = graph.enqueue::<embedding_rows::Entry>(
        embedding,
        &dimensions,
        embedding_rows::WorkflowArgs {
            table: table.tensor().into(),
            tokens: tokens.tensor().into(),
            directory: directory.tensor().into(),
            scale,
            normalize,
            epsilon,
        },
    )?;
    graph.export(&result.r1)?;
    Ok(EntryGraphDraft {
        graph,
        table,
        directory,
        tokens,
        hidden: result.r1,
    })
}

/// `embedding_rows`' (`scale`, `normalize`, `epsilon`) for the decoder's
/// entry form: the text-row scale, and the unweighted RMS when present (its
/// epsilon is not read otherwise).
pub(crate) fn embedding_transform(geometry: &Decoder) -> (f32, i32, f32) {
    let scale = match geometry.entry.scale {
        EmbeddingScale::Unit => 1.0,
        EmbeddingScale::SqrtHidden => (geometry.hidden as f32).sqrt(),
    };
    match geometry.entry.norm {
        Some(norm) => (scale, 1, norm.epsilon as f32),
        None => (scale, 0, 0.0),
    }
}

#[cfg(test)]
pub(crate) fn checked_entry_graph_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    rows: u64,
    uploaded: bool,
) -> Result<NativeGraphStorageBytes, GraphError> {
    let weight = embedding_weight(load)?;
    let elements = [("EW", weight.resident), ("A", activation(geometry))];
    let draft = entry_graph_topology(
        NativeGraphMetadata::new(backend),
        &elements,
        weight,
        geometry,
        rows,
        if uploaded {
            EntryTokens::Uploaded
        } else {
            EntryTokens::Selected
        },
    )?;
    GraphDraft::seal(draft.graph)
}

#[derive(Clone)]
pub(crate) struct PreparedTargetBlockGraph {
    pub plan: NativeGraphPlan,
    pub hidden: NativePort,
    pub weights: Vec<(WeightPort, NativePort)>,
    /// Host constants bound statically with the weights.
    pub constants: Vec<GraphConstant>,
    pub state: BlockStatePorts,
    pub controls: BlockControlPorts,
    /// The draft tap of a tapped block.
    pub taps: Option<TapPorts>,
    /// The per-layer rows port of a block with a per-layer input sublayer.
    pub per_layer: Option<NativePort>,
    pub output: WorkflowTensor,
}

/// What a graph's weight port binds of the resident weight of `role`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct WeightPort {
    pub role: WeightRole,
    pub part: WeightPart,
}

/// A resident weight's bound parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WeightPart {
    /// The weight in its resident representation.
    Values,
    /// Its resident second-level scale (`WeightPlan::scale`).
    Scale,
    /// Its resident packing bound (`WeightPlan::bound`).
    Bound,
    /// A streamed gathered table's directory of every row's slot
    /// (`WeightPlan::streamed`).
    Directory,
}

impl WeightPart {
    /// This part of `weight`.
    pub(crate) fn of(self, weight: &crate::ResidentWeight) -> Result<&seismic::Tensor, String> {
        match self {
            Self::Values => Ok(weight.tensor()),
            Self::Directory => weight.table().directory().ok_or_else(|| {
                format!(
                    "resident weight {:?} is held whole and has no directory",
                    weight.descriptor().name
                )
            }),
            Self::Scale => weight.scale().ok_or_else(|| {
                format!(
                    "resident weight {:?} has no second-level scale",
                    weight.descriptor().name
                )
            }),
            Self::Bound => weight.bound().ok_or_else(|| {
                format!(
                    "resident weight {:?} has no packing bound",
                    weight.descriptor().name
                )
            }),
        }
    }
}

/// The port of the planned weight of `role`'s `part`.
fn weight_port<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    role: WeightRole,
    part: WeightPart,
    ports: &mut Vec<(WeightPort, NativePort)>,
) -> Result<NativePort, GraphError> {
    let plan = load
        .weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| format!("missing planned weight {role:?}"))?;
    let port = match (part, &plan.scale) {
        (WeightPart::Values, _) => graph.port(plan.resident, &plan.shape),
        (WeightPart::Scale, Some(scale)) => graph.port(Element::f32(), &[scale.extent]),
        (WeightPart::Scale, None) => {
            return Err(format!("planned weight {role:?} has no second-level scale").into());
        }
        (WeightPart::Bound, _) if plan.bound != 0 => graph.port(Element::f32(), &[plan.bound]),
        (WeightPart::Bound, _) => {
            return Err(format!("planned weight {role:?} has no packing bound").into());
        }
        (WeightPart::Directory, _) => match plan.streamed {
            Some(streamed) => graph.port(Element::i32(), &[streamed.table_rows]),
            None => {
                return Err(
                    format!("planned weight {role:?} is held whole and has no directory").into(),
                );
            }
        },
    }?;
    ports.push((WeightPort { role, part }, port.clone()));
    Ok(port)
}

/// What a reader of a gathered table addresses its rows through
/// (`gathered_row`): a directory of every row's slot, or an empty one under
/// which the reader reads each row it is given.
#[derive(Clone)]
pub(crate) struct RowDirectory {
    pub tensor: WorkflowTensor,
    /// Its entries: the readers' `R`.
    pub entries: u64,
}

impl RowDirectory {
    /// The empty directory, a constant of the graph.
    pub(crate) fn direct<G: GraphDraft>(
        graph: &mut G,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let directory = GraphConstant::i32(graph, &[])?;
        let tensor = directory.port().tensor().clone();
        constants.push(directory);
        Ok(Self { tensor, entries: 0 })
    }
}

/// A gathered table as a graph's reads find it: the tensor they gather from
/// (the table's rows, or the slots of a streamed one) and the directory they
/// address it through. A read of rows the host knows gathers by the indices
/// the host writes (`ResidentTable::indices`): `by_rows`. A read of rows the
/// device selected finds their slots in the table's directory: `selected`.
pub(crate) struct GatheredTable {
    pub table: WorkflowTensor,
    /// The rows of `table`: the readers' `V`.
    pub rows: u64,
    role: WeightRole,
    /// The entries of a streamed table's directory; 0 of a table held whole.
    entries: u64,
    /// The empty directory, declared by the graph's first read by rows.
    direct: Option<RowDirectory>,
    /// The directory's port, declared by the graph's first read of selected
    /// rows: a graph that reads none has no such port.
    directory: Option<RowDirectory>,
}

impl GatheredTable {
    /// The planned table of `scope` and `kind`.
    pub(crate) fn of<G: GraphDraft>(
        graph: &mut G,
        load: &ModelLoadPlan,
        scope: WeightScope,
        kind: WeightKind,
        ports: &mut Vec<(WeightPort, NativePort)>,
    ) -> Result<Self, GraphError> {
        let role = WeightRole { scope, kind };
        let plan = load
            .weights()
            .find(|plan| plan.role == role)
            .ok_or_else(|| format!("missing planned weight {role:?}"))?;
        let rows = plan.shape[0];
        let table = weight_port(graph, load, role, WeightPart::Values, ports)?
            .tensor()
            .clone();
        Ok(Self {
            table,
            rows,
            role,
            entries: plan.directory_entries(),
            direct: None,
            directory: None,
        })
    }

    /// The directory of a read of the table's own rows, as the host gives
    /// them: empty.
    pub(crate) fn by_rows<G: GraphDraft>(
        &mut self,
        graph: &mut G,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<RowDirectory, GraphError> {
        if let Some(direct) = &self.direct {
            return Ok(direct.clone());
        }
        Ok(self
            .direct
            .insert(RowDirectory::direct(graph, constants)?)
            .clone())
    }

    /// The directory of a read of rows the device selected: empty of a table
    /// held whole, a streamed table's directory of every row's slot.
    pub(crate) fn selected<G: GraphDraft>(
        &mut self,
        graph: &mut G,
        load: &ModelLoadPlan,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<RowDirectory, GraphError> {
        if self.entries == 0 {
            return self.by_rows(graph, constants);
        }
        if let Some(directory) = &self.directory {
            return Ok(directory.clone());
        }
        let tensor = weight_port(graph, load, self.role, WeightPart::Directory, ports)?
            .tensor()
            .clone();
        Ok(self
            .directory
            .insert(RowDirectory {
                tensor,
                entries: self.entries,
            })
            .clone())
    }
}

/// The port of a weight an entry binds without an accumulator-scale port
/// (plan admission refused a scaled one: `PlanError::UnportedScale`).
pub(crate) fn weight<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
) -> Result<seismic::WorkflowTensor, GraphError> {
    weight_port(
        graph,
        load,
        WeightRole { scope, kind },
        WeightPart::Values,
        ports,
    )
    .map(|port| port.tensor().clone())
}

/// A projection weight and the tensor its entry's accumulator-scale port
/// binds: its resident second-level scale, else the graph's absent operand
/// (`GraphConstant::absent`: zero extent, never read), and that port's
/// static extent.
pub(crate) struct ScaledWeight {
    pub weight: WorkflowTensor,
    pub scale: WorkflowTensor,
    pub extent: u64,
}

impl ScaledWeight {
    /// A weight bound with the absent scale `absent` (zero extent).
    pub(crate) fn unscaled(weight: WorkflowTensor, absent: WorkflowTensor) -> Self {
        Self {
            weight,
            scale: absent,
            extent: 0,
        }
    }
}

/// A weight's packing bound as the bound port of an entry with a packed
/// form binds it (`WeightPlan::bound`, `[B] f32`): its resident bound and
/// the B runs it holds, or the graph's absent operand at B = 0 where the
/// weight has none.
#[derive(Clone)]
pub(crate) struct PackingBound {
    pub tensor: WorkflowTensor,
    pub extent: u64,
}

impl PackingBound {
    /// The packing bound of the planned weight of `role`.
    pub(crate) fn planned<G: GraphDraft>(
        graph: &mut G,
        load: &ModelLoadPlan,
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let extent = load
            .weights()
            .find(|plan| plan.role == role)
            .map(|plan| plan.bound)
            .ok_or_else(|| format!("missing planned weight {role:?}"))?;
        match extent {
            0 => Self::absent(graph, constants),
            _ => Ok(Self {
                tensor: weight_port(graph, load, role, WeightPart::Bound, ports)?
                    .tensor()
                    .clone(),
                extent,
            }),
        }
    }

    /// No bound: the graph's absent operand.
    pub(crate) fn absent<G: GraphDraft>(
        graph: &mut G,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            tensor: GraphConstant::absent(graph, constants)?.tensor().clone(),
            extent: 0,
        })
    }
}

/// A projection weight an entry with a packed form binds: its scaled operand
/// and its packing bound.
pub(crate) struct PackedWeight {
    pub scaled: ScaledWeight,
    pub bound: PackingBound,
}

/// Register one scope-owned readout representation and its scale lifetime.
pub(crate) fn projection_ports<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    certified: bool,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<super::graph::readout::ProjectionPorts, GraphError> {
    use super::graph::readout::{HeadPlans, ProjectionPorts};
    Ok(match HeadPlans::of(load, scope)?.selected() {
        HeadPlans::Packed(_) => {
            let (weight, scale, scale_extent) =
                scaled_weight_ports(graph, load, scope, WeightKind::Output, ports, constants)?;
            ProjectionPorts::Packed {
                weight,
                scale,
                scale_extent,
            }
        }
        plans @ HeadPlans::Progressive(_) => {
            let depth = plans.depth().ok_or("progressive projection has no depth")?;
            let planes = magnitude_family_contracts::ProgressivePlane::ALL
                .into_iter()
                .filter(|plane| {
                    certified || *plane != magnitude_family_contracts::ProgressivePlane::Radius
                })
                .map(|plane| {
                    Ok((
                        plane,
                        weight_port(
                            graph,
                            load,
                            WeightRole {
                                scope,
                                kind: WeightKind::OutputPlane(plane),
                            },
                            WeightPart::Values,
                            ports,
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>, GraphError>>()?;
            ProjectionPorts::Progressive { depth, planes }
        }
        HeadPlans::Qualified { .. } => unreachable!("a recipe was selected above"),
    })
}

/// The port of a weight an entry binds with a per-tensor accumulator-scale
/// port (`[S] f32`, S ∈ {0, 1}).
pub(crate) fn scaled_weight<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<ScaledWeight, GraphError> {
    let (weight, scale, extent) = scaled_weight_ports(graph, load, scope, kind, ports, constants)?;
    Ok(ScaledWeight {
        weight: weight.tensor().clone(),
        scale: scale.tensor().clone(),
        extent,
    })
}

/// [`scaled_weight`] of an entry with a packed form, with the weight's
/// packing bound.
pub(crate) fn packed_weight<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<PackedWeight, GraphError> {
    Ok(PackedWeight {
        scaled: scaled_weight(graph, load, scope, kind, ports, constants)?,
        bound: PackingBound::planned(graph, load, WeightRole { scope, kind }, ports, constants)?,
    })
}

/// The same scope-owned scaled operand as graph ports, for common projection
/// builders. Registration and absent-scale lifetime have one authority.
pub(crate) fn scaled_weight_ports<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<(NativePort, NativePort, u64), GraphError> {
    let role = WeightRole { scope, kind };
    let weight = weight_port(graph, load, role, WeightPart::Values, ports)?;
    let (scale, extent) = weight_scale_port(graph, load, role, ports, constants)?;
    Ok((weight, scale, extent))
}

/// Register the accumulator-scale operand, including its absent lifetime.
pub(crate) fn weight_scale_port<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    role: WeightRole,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<(NativePort, u64), GraphError> {
    let extent = planned_scale_extent(load, role)?;
    let scale = match extent {
        0 => GraphConstant::absent(graph, constants)?,
        1 => weight_port(graph, load, role, WeightPart::Scale, ports)?,
        _ => return Err(format!("{role:?} has a per-matrix scale at a per-tensor port").into()),
    };
    Ok((scale, extent))
}

/// The port of a weight's resident second-level scale (`[1] f32`, or
/// `[E] f32` for a stacked expert weight), or `None` when it has none.
pub(crate) fn resident_scale<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
) -> Result<Option<WorkflowTensor>, GraphError> {
    let role = WeightRole { scope, kind };
    match planned_scale_extent(load, role)? {
        0 => Ok(None),
        _ => weight_port(graph, load, role, WeightPart::Scale, ports)
            .map(|port| Some(port.tensor().clone())),
    }
}

fn planned_scale_extent(load: &ModelLoadPlan, role: WeightRole) -> Result<u64, String> {
    load.weights()
        .find(|plan| plan.role == role)
        .map(crate::WeightPlan::scale_extent)
        .ok_or_else(|| format!("missing planned weight {role:?}"))
}

pub(crate) fn activation(geometry: &Decoder) -> Element {
    match geometry.activation_dtype {
        magnitude_family_contracts::ActivationDType::F16 => Element::f16(),
        magnitude_family_contracts::ActivationDType::BF16 => Element::bf16(),
    }
}

impl PreparedTargetBlockGraph {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<'a>(
        device: &Device,
        handle: &'a AttestedTargetBlock,
        tap: Option<TapEntry<'a, seismic::NativeGraph>>,
        per_layer: Option<crate::PerLayerBinding>,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        block_index: usize,
        rows: u64,
        segments: u64,
        slots: u64,
        listed: u64,
        layout: &NativeGraphLayout,
        family: &FamilyConstants,
    ) -> Result<Self, String> {
        let mut graph = device.native_graph_with_layout(layout);
        let mut weights = Vec::new();
        let mut constants = Vec::new();
        let hidden = graph
            .port(Element::f32(), &[rows, geometry.hidden])
            .map_err(|error| error.to_string())?;
        let mixer = (&handle.mixer).into();
        let feed_forward = handle.feed_forward.as_ref().map(Into::into);
        let per_layer = match (&handle.per_layer, per_layer) {
            (None, None) => None,
            (Some(kernels), Some(binding)) => Some((kernels.into(), binding)),
            _ => return Err("the block's per-layer kernels disagree with its binding".into()),
        };
        let parts = block_graph(
            &mut graph,
            tap,
            mixer,
            feed_forward,
            per_layer,
            hidden.tensor(),
            BlockGraphInputs {
                load,
                geometry,
                state,
                block_index,
                rows,
                segments,
                slots,
                listed,
                output_scales: handle.output_scales,
            },
            &mut weights,
            &mut constants,
        )
        .map_err(|error| error.to_string())?;
        graph
            .export(&parts.output)
            .map_err(|error| error.to_string())?;
        let plan = graph.seal().map_err(|error| error.to_string())?;
        // The graph's constants are the family's values, and its vectors
        // hold exactly their elements: what it owns then follows from its
        // topology (`owned_vectors_heap`).
        for constant in &mut constants {
            family.share(constant);
        }
        let mut state = parts.state;
        weights.shrink_to_fit();
        constants.shrink_to_fit();
        state.shrink_to_fit();
        Ok(Self {
            plan,
            hidden,
            weights,
            constants,
            state,
            controls: parts.controls,
            taps: parts.taps,
            per_layer: parts.per_layer,
            output: parts.output,
        })
    }

    /// The host heap of the vectors this graph owns, at their capacities.
    /// Its plan is counted apart (`NativeGraphPlan::host_heap`), and its
    /// constants' values are the family's (`FamilyConstants`).
    fn owned_heap(&self) -> HostHeap {
        owned_vectors_heap(
            self.weights.capacity(),
            self.constants.capacity(),
            self.state.ports().capacity(),
        )
    }
}

/// Everything a block graph is built from apart from its kernel entries.
struct BlockGraphInputs<'a> {
    load: &'a ModelLoadPlan,
    geometry: &'a Decoder,
    state: &'a StateResourcePlan,
    block_index: usize,
    rows: u64,
    segments: u64,
    slots: u64,
    listed: u64,
    output_scales: OutputScales,
}

struct BlockGraphParts {
    output: WorkflowTensor,
    state: BlockStatePorts,
    controls: BlockControlPorts,
    taps: Option<TapPorts>,
    /// The per-layer rows a block with a per-layer input sublayer reads.
    per_layer: Option<NativePort>,
}

/// One decoder block's graph fragment: its draft tap when tapped, then its
/// mixer then its feed-forward over `hidden`. The production graph and the
/// checked metadata route both build blocks through this one function.
#[allow(clippy::too_many_arguments)]
fn block_graph<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    tap_entry: Option<TapEntry<'a, G>>,
    mixer_entries: MixerEntries<'a, G>,
    feed_forward_entries: Option<FeedForwardEntries<'a, G>>,
    per_layer_entries: Option<PerLayerParts<'a, G>>,
    hidden: &WorkflowTensor,
    inputs: BlockGraphInputs<'_>,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<BlockGraphParts, GraphError> {
    let BlockGraphInputs {
        load,
        geometry,
        state,
        block_index,
        rows,
        segments,
        slots,
        listed,
        output_scales,
    } = inputs;
    let block = geometry
        .blocks
        .get(block_index)
        .ok_or("target block geometry is absent")?;
    let paired = paired_block(block).map_err(|error| error.to_string())?;
    let sublayers = block::BlockSublayers {
        lane: block::BlockLane::Target,
        paired: &paired,
        load,
        geometry,
        state,
        block_index,
        rows,
        segments,
        slots,
        listed,
        output_scales,
    };
    let mut taps = tap_entry
        .map(|entry| Taps::new(graph, entry, rows, geometry.hidden, activation(geometry)))
        .transpose()?;
    if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.entry) {
        taps.ports.entry = Some(taps.tap(graph, hidden)?);
    }
    let (mixed, state_ports, controls) =
        sublayers.mixer(graph, mixer_entries, hidden, weights, constants)?;
    if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.middle) {
        if paired.feed_forward.is_none() {
            return Err("a draft taps the feed-forward of a lone mixer block".into());
        }
        taps.ports.middle = Some(taps.tap(graph, &mixed)?);
    }
    let output = sublayers.feed_forward(graph, feed_forward_entries, mixed, weights, constants)?;
    let (output, per_layer) =
        sublayers.per_layer(graph, per_layer_entries, output, weights, constants)?;
    if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.output) {
        taps.ports.output = Some(taps.tap(graph, &output)?);
    }
    Ok(BlockGraphParts {
        output,
        state: state_ports,
        controls,
        taps: taps.map(|taps| taps.ports),
        per_layer,
    })
}

/// Check the production block topology from its planned entry bindings
/// without forming kernels or allocating device storage: the graph, its
/// constants, and the host heap of the vectors its prepared graph owns
/// (`owned_vectors_heap`).
#[allow(clippy::too_many_arguments)]
fn checked_block_graph_draft(
    mut graph: NativeGraphMetadata,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    slot: TargetBlockProgramSlot,
    tap: Option<(u64, TapPositions)>,
    block_index: usize,
    rows: u64,
    segments: u64,
    slots: u64,
    listed: u64,
) -> Result<(NativeGraphMetadata, Vec<GraphConstant>, HostHeap), GraphError> {
    let mut weights = Vec::new();
    let mut constants = Vec::new();
    let tap_elements = [("A", activation(geometry))];
    let tap_entry = tap.map(|(width, positions)| TapEntry {
        entry: &tap_elements[..],
        width,
        positions,
    });
    let hidden = GraphDraft::port_with_class_extent(
        &mut graph,
        Element::f32(),
        &[rows, geometry.hidden],
        0,
        "M",
    )?;
    let checked_mixer = block::CheckedMixerEntries::new(
        slot.mixer(),
        graph.backend(),
        slot_lists(state, graph.backend(), slot),
    )?;
    let mixer = checked_mixer.entries()?;
    let checked_feed_forward = slot
        .feed_forward()
        .map(|slot| block::CheckedFeedForwardEntries::new(slot, graph.backend()))
        .transpose()?;
    let feed_forward = checked_feed_forward
        .as_ref()
        .map(|checked| checked.entries());
    let checked_per_layer = block::checked_per_layer(slot.per_layer())?;
    let per_layer = checked_per_layer
        .as_ref()
        .map(|(checked, binding)| (checked.entries(), *binding));
    let parts = block_graph(
        &mut graph,
        tap_entry,
        mixer,
        feed_forward,
        per_layer,
        hidden.tensor(),
        BlockGraphInputs {
            load,
            geometry,
            state,
            block_index,
            rows,
            segments,
            slots,
            listed,
            // Scalar arguments leave the checked shapes and storage alone.
            output_scales: OutputScales {
                mixer: 1.0,
                feed_forward: 1.0,
                per_layer: 1.0,
            },
        },
        &mut weights,
        &mut constants,
    )?;
    GraphDraft::export(&mut graph, &parts.output)?;
    let owned = owned_vectors_heap(weights.len(), constants.len(), parts.state.ports().len());
    Ok((graph, constants, owned))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn checked_block_graph_resources(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    slot: TargetBlockProgramSlot,
    block_index: usize,
    rows: u64,
    segments: u64,
    slots: u64,
    listed: u64,
) -> Result<(NativeGraphStorageBytes, Vec<GraphConstant>), GraphError> {
    let (graph, constants, _) = checked_block_graph_draft(
        NativeGraphMetadata::new(backend),
        load,
        geometry,
        state,
        slot,
        None,
        block_index,
        rows,
        segments,
        slots,
        listed,
    )?;
    Ok((GraphDraft::seal(graph)?, constants))
}

/// The same decoder class ladder as `PreparedTargetGraphs::prepare`. The
/// family takes independent maxima because its workspace, output and upload
/// holdings may peak in different members.
#[cfg(test)]
pub(crate) fn checked_target_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<CheckedGraphResources, GraphError> {
    certify_target_family(backend, load, geometry, state, plan, limits)
        .map(|certificate| certificate.resources)
}

/// The family's storage with the graphs a load of it seals and the host
/// heap they hold: what the load's assessment charges, from the one
/// enumeration the sealer follows.
pub(crate) fn checked_target_family(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<(CheckedGraphResources, PlannedSealedGraphs), GraphError> {
    certify_target_family(backend, load, geometry, state, plan, limits)
        .map(|certificate| (certificate.resources, certificate.sealed))
}

struct TargetFamilyCertificate {
    resources: CheckedGraphResources,
    /// The graphs a load seals into these layouts, and their host heap.
    sealed: PlannedSealedGraphs,
    entries: BTreeMap<EntryTokens, NativeGraphLayout>,
    /// The per-layer entry graph's layout, of a per-layer entry.
    per_layer: Option<NativeGraphLayout>,
    /// Per block, what each row regime's classes seal into, those that
    /// list their launch's history row tiles apart.
    blocks: Vec<BTreeMap<(RowForm, bool), CertifiedBlockGraph>>,
}

/// What every class of one block's row regime prepares to: one topology.
#[derive(Clone)]
struct CertifiedBlockGraph {
    /// The layout its plans seal into.
    layout: NativeGraphLayout,
    /// The host heap of the vectors each of its prepared graphs owns
    /// (`owned_vectors_heap`). Every weight, constant and state port is a
    /// port of the graph, and the layout holds its classes to its ports,
    /// so the template's lengths are each class's.
    owned: HostHeap,
}

fn certify_target_family(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<TargetFamilyCertificate, GraphError> {
    if plan.blocks().len() != geometry.blocks.len() {
        return Err("planned target block count disagrees with geometry".into());
    }
    let max_slots = u64::try_from(limits.max_launch_slots)
        .map_err(|_| "target request slot bound exceeds u64")?;
    let max_segments = u64::try_from(
        state
            .target_state()
            .span_limit()
            .checked_next_power_of_two()
            .ok_or("target segment class overflows")?,
    )
    .map_err(|_| "target segment class exceeds u64")?;
    let (tap_width, tapped) = block_taps(plan, geometry)?;
    let shapes = block_graph_shapes(load, geometry, state, &tapped)?;
    let distinct_blocks = (0..shapes.len())
        .filter(|&index| !(0..index).any(|earlier| shapes[earlier] == shapes[index]))
        .collect::<Vec<_>>();
    let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
        .into_iter()
        .map(|rows| rows as u64)
        .collect::<Vec<_>>();
    if row_classes.is_empty() {
        return Err("target batch row bound has no row class".into());
    }
    let mut family = CheckedGraphFamilyResources::new();
    let mut entries = BTreeMap::new();
    let largest_rows = *row_classes.last().expect("row classes are nonempty");
    for source in [EntryTokens::Uploaded, EntryTokens::Selected] {
        let weight = embedding_weight(load)?;
        let elements = [("EW", weight.resident), ("A", activation(geometry))];
        let graph = entry_graph_topology(
            NativeGraphMetadata::new_template(backend),
            &elements,
            weight,
            geometry,
            largest_rows,
            source,
        )?
        .graph;
        let layout = graph
            .seal_template()
            .and_then(|template| {
                template.certify(&[
                    NativeGraphClassSlice::new().dimension("M", row_classes.iter().copied())
                ])
            })
            .map_err(|error| format!("target entry graph: {error}"))?;
        family.include(layout.storage_bytes(), []);
        entries.insert(source, layout);
    }
    let per_layer = match (plan.per_layer(), &geometry.entry.per_layer) {
        (None, None) => None,
        (Some(binding), Some(entry)) => {
            let checked = CheckedPerLayerEntryEntries::new(binding);
            let mut graph = NativeGraphMetadata::new_template(backend);
            per_layer_entry(
                &mut graph,
                checked.entries(),
                target_weight(load, WeightKind::PerLayerModelProjection)?,
                target_weight(load, WeightKind::PerLayerProjectionNorm)?,
                binding,
                entry,
                largest_rows,
            )?;
            let layout = graph
                .seal_template()
                .and_then(|template| {
                    template.certify(&[per_layer_entry_class_slice(row_classes.iter().copied())])
                })
                .map_err(|error| format!("per-layer entry graph: {error}"))?;
            family.include(layout.storage_bytes(), []);
            Some(layout)
        }
        _ => return Err("the per-layer entry's binding and geometry disagree".into()),
    };
    let mut blocks = vec![BTreeMap::new(); geometry.blocks.len()];
    for &index in &distinct_blocks {
        let slot = plan.blocks()[index];
        let paired = paired_block(&geometry.blocks[index]).map_err(|error| error.to_string())?;
        let mut regimes: BTreeMap<RowForm, Vec<u64>> = BTreeMap::new();
        for &rows in &row_classes {
            regimes.entry(RowForm::of(rows)).or_default().push(rows);
        }
        for (form, rows) in regimes {
            let largest = *rows.last().expect("a regime holds a row class");
            // A regime's listing classes are their own topology (another
            // kernel of the entry, and the list's input), over every listed
            // tile count.
            for listed in [false, true] {
                let listings = match listed {
                    false => Vec::new(),
                    true => block_listings(state, backend, slot, index, largest)?.tiles,
                };
                let mut listing_rows = Vec::new();
                for &rows in &rows {
                    if !listed
                        || !block_listings(state, backend, slot, index, rows)?
                            .tiles
                            .is_empty()
                    {
                        listing_rows.push(rows);
                    }
                }
                let rows = listing_rows;
                if rows.is_empty() {
                    continue;
                }
                let (graph, constants, owned) = checked_block_graph_draft(
                    NativeGraphMetadata::new_template(backend),
                    load,
                    geometry,
                    state,
                    slot,
                    tap_width
                        .map(|width| (width, tapped[index].positions()))
                        .filter(|(_, positions)| positions.any()),
                    index,
                    largest,
                    1,
                    1,
                    listings.last().copied().unwrap_or(0),
                )?;
                let slices = rows
                    .iter()
                    .map(|&rows| {
                        block_class_slice(&paired, rows, max_slots, max_segments, &listings)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let layout = graph
                    .seal_template()
                    .and_then(|template| template.certify(&slices))
                    .map_err(|error| format!("target block {index}: {error}"))?;
                family.include(layout.storage_bytes(), constants);
                blocks[index].insert((form, listed), CertifiedBlockGraph { layout, owned });
            }
        }
        // Row-class constants (`block::class_constants_of`).
        for &rows in &row_classes {
            let mut constants = block::class_constants_of(&paired, geometry.hidden, rows)?;
            // Causal K8/V4 decode publishes fresh rows first, then disables
            // the fused append with a row-sized destination table. A template
            // carries only its largest row class's table; charge every class.
            if let crate::MixerProgramSlot::Attention(binding) = slot.mixer() {
                if binding.history.encoding() == magnitude_state::HistoryEncoding::Affine
                    && binding.shape.fresh > 0
                    && rows > 1
                    && operators::attention::graph::decodes(rows)
                {
                    constants.push(GraphConstant::i32_value(&vec![-1; rows as usize]));
                }
            }
            family.include(
                NativeGraphStorageBytes {
                    workspace: 0,
                    output: 0,
                    upload: 0,
                },
                constants,
            );
        }
    }
    for index in 0..blocks.len() {
        if let Some(first) = (0..index).find(|&first| shapes[first] == shapes[index]) {
            blocks[index] = blocks[first].clone();
        }
    }
    // The graphs a load seals: two entries and the per-layer entry per row
    // class, and the enumerated block plans, each into its regime's layout.
    // Layer scales the plan does not know are taken both ways, which
    // bounds the sealed block plans from both sides.
    let mixers = block_mixers(geometry)?;
    let enumerate = |unread| {
        enumerate_block_graphs(
            backend,
            state,
            plan,
            &mixers,
            &shapes,
            BlockOutputScales::Planned(unread),
            &row_classes,
            max_slots,
            max_segments,
        )
    };
    let enumeration = enumerate(UnreadScales::Distinct)?;
    let least = match enumeration.exact {
        true => None,
        false => Some(enumerate(UnreadScales::Equal)?),
    };
    let count = |sealing: &[(usize, BlockClass)]| {
        let mut planned = PlannedHeap::default();
        for _ in &row_classes {
            for layout in entries.values().chain(&per_layer) {
                planned.seal(layout)?;
            }
        }
        for &(index, (rows, _, _, listed)) in sealing {
            planned.seal_block(
                blocks[index]
                    .get(&(RowForm::of(rows), listed > 0))
                    .ok_or("a block class has no certified layout")?,
            )?;
        }
        Ok::<_, GraphError>((planned, sealing.len()))
    };
    let (planned, sealed_blocks) = count(&enumeration.sealing)?;
    // The bindings a load makes of them: one for each entry graph, and one
    // for every block's every class, whichever block sealed its plan.
    let mut bound = HostHeap::default();
    let mut binding = |layout: &NativeGraphLayout| {
        bound = bound
            .plus(layout.bound_heap())
            .ok_or("planned graph binding heap overflows")?;
        Ok::<_, GraphError>(())
    };
    for _ in &row_classes {
        for layout in entries.values().chain(&per_layer) {
            binding(layout)?;
        }
    }
    let mut block_bindings = vec![0usize; blocks.len()];
    let bound_classes = enumeration.sealing.iter().copied().chain(
        enumeration
            .sharing
            .iter()
            .map(|&(index, class, _)| (index, class)),
    );
    for (index, (rows, _, _, listed)) in bound_classes {
        binding(
            &blocks[index]
                .get(&(RowForm::of(rows), listed > 0))
                .ok_or("a block class has no certified layout")?
                .layout,
        )?;
        block_bindings[index] += 1;
    }
    let bound = bound
        .plus(block_binding_storage(&block_bindings)?)
        .ok_or("planned graph binding heap overflows")?;
    // Every block holds a graph of each of its classes, whichever block
    // sealed it; what differs with the count is the sealed graphs' shared
    // allocations.
    let graphs = |(planned, sealed_blocks): (PlannedHeap, usize)| {
        Ok::<_, GraphError>(SealedGraphsHeap {
            graphs: planned.graphs,
            plans: planned.plans,
            storage: block_graph_storage(&block_bindings, sealed_blocks)?,
            owned: planned.owned,
        })
    };
    let at_most = graphs((planned, sealed_blocks))?;
    let at_least = match &least {
        None => at_most,
        Some(least) => graphs(count(&least.sealing)?)?,
    };
    let sealed = PlannedSealedGraphs {
        at_most,
        at_least,
        bound,
        constants: family.constant_heap()?,
    };
    // The draft input rows and the per-layer rows the family binds with its
    // constants.
    let mut resources = family.finish()?;
    let per_layer_rows = per_layer_rows_bytes(per_layer_rows(plan, largest_rows))?;
    resources.binding_constant_bytes = resources
        .binding_constant_bytes
        .checked_add(tap_buffer_bytes(tap_width, largest_rows, geometry)?)
        .and_then(|bytes| bytes.checked_add(per_layer_rows))
        .ok_or("target binding charge overflows")?;
    Ok(TargetFamilyCertificate {
        resources,
        sealed,
        entries,
        per_layer,
        blocks,
    })
}

/// Bytes of the draft input rows a tapped target binds: `[rows, width]`
/// activations over the largest row class, which every class binds a
/// leading slice of.
fn tap_buffer_bytes(width: Option<u64>, rows: u64, geometry: &Decoder) -> Result<u64, String> {
    width.map_or(Ok(0), |width| {
        activation(geometry)
            .canonical_byte_len(&[rows, width])
            .map_err(|error| error.to_string())
    })
}

#[cfg(test)]
mod stateless_tail_tests {
    use super::trailing_run;

    #[test]
    fn the_tail_is_the_trailing_run_of_stateless_blocks() {
        // Gemma 4 E2B: 35 blocks, the last 20 read shared history.
        assert_eq!(trailing_run(35, |index| index >= 15), 15);
        // A stateful block inside the run ends it.
        assert_eq!(trailing_run(35, |index| index >= 15 && index != 30), 31);
        // A stateful last block leaves no tail; nor do stateless leading blocks.
        assert_eq!(trailing_run(35, |index| index < 34), 35);
        assert_eq!(trailing_run(4, |_| false), 4);
        assert_eq!(trailing_run(4, |_| true), 0);
    }
}

#[cfg(test)]
mod resource_template_tests {
    use super::*;
    use crate::{
        ComponentSelection, ExecutionPath, PlannedMethod, ResourceCapacity, ResourcePlanner,
        assessment::fixtures::{QWEN35_CONFIGURATIONS, declared_model},
        resident_layout,
    };
    use magnitude_state::KvCodec;

    /// The 512-row attention class workspaces of one model on Metal over
    /// K8/V4 history: the family's, each listing class's and that of the class
    /// that lists none, with the store's reservation of history rows.
    struct Charged {
        reservation: u64,
        family: u64,
        listed: Vec<u64>,
        unlisted: u64,
    }

    fn charged(
        (kv_heads, head_width): (u64, u64),
        context: u64,
        domain_bytes: u64,
        tensor_operations: bool,
    ) -> Charged {
        let limits = ResourceLimits {
            max_launch_rows: 512,
            max_launch_slots: 512,
            max_selected_rows: 64,
            max_drafting_slots: 64,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
        };
        let mut configuration = QWEN35_CONFIGURATIONS[0];
        configuration.blocks = configuration.attention_interval;
        configuration.kv_heads = kv_heads;
        configuration.head_width = head_width;
        let (mut definition, manifest) = declared_model(&configuration);
        definition.decoder.context_limit = context;
        let backend = BackendName::Metal;
        let codec = KvCodec::AffineK8V4;
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            resident_layout(ExecutionPath::Native, backend),
        )
        .unwrap();
        let plan = load.program_plan(&definition, codec).unwrap();
        let index = plan
            .target()
            .blocks()
            .iter()
            .position(|slot| matches!(slot.mixer(), crate::MixerProgramSlot::Attention(_)))
            .unwrap();
        let state = ResourcePlanner::state_plan(
            &definition,
            &load,
            PlannedMethod::Plain,
            codec,
            limits,
            ResourceCapacity {
                domain_bytes,
                tensor_operations: crate::TensorOperations::of(tensor_operations),
            },
        )
        .unwrap();
        let certificate = certify_target_family(
            backend,
            &load,
            &definition.decoder,
            &state,
            plan.target(),
            limits,
        )
        .unwrap();
        let family = certificate.resources.storage.workspace;
        // The load's assessment charges the family's workspace, and every
        // exact 512-row class of the block seals into the layout the family
        // certified for it and is charged that layout.
        let assessed = crate::AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            PlannedMethod::Plain,
            codec,
            limits,
            backend,
        )
        .unwrap();
        assert_eq!(assessed.target.workspace_bytes, family);
        let max_segments = state.target_state().span_limit().next_power_of_two() as u64;
        let listings =
            block_listings(&state, backend, plan.target().blocks()[index], index, 512).unwrap();
        // Tensor operations read every head's history decoded; without
        // them the co-issue form reads 128-, 256- and 512-column heads'.
        assert_eq!(
            listings.tiles.is_empty(),
            !(tensor_operations || matches!(head_width, 128 | 256 | 512))
        );
        for (rows, segments, slots, listed) in
            block_classes(MixerKind::Attention, 512, 512, max_segments, &listings)
        {
            let (graph, _, _) = checked_block_graph_draft(
                NativeGraphMetadata::new(backend),
                &load,
                &definition.decoder,
                &state,
                plan.target().blocks()[index],
                None,
                index,
                rows,
                segments,
                slots,
                listed,
            )
            .unwrap();
            let layout = &certificate.blocks[index][&(RowForm::of(rows), listed > 0)].layout;
            assert_eq!(
                graph.seal_with_layout(layout).unwrap(),
                layout.storage_bytes()
            );
            assert!(layout.storage_bytes().workspace <= family);
        }
        let class = |listed| {
            checked_block_graph_resources(
                backend,
                &load,
                &definition.decoder,
                &state,
                plan.target().blocks()[index],
                index,
                512,
                1,
                1,
                listed,
            )
            .unwrap()
            .0
            .workspace
        };
        // One request's worth of tiles is the largest listing: its context
        // and the rows appended within one advance (a causal reader of
        // compressed history reads them through their reserved
        // destinations), each in at most two partial pages.
        let history = state
            .target_state()
            .layer_history(LayerRef::Target(index as u32))
            .unwrap()
            .store;
        if let Some(&most) = listings.tiles.last() {
            assert!(
                most * 256
                    <= context + limits.max_launch_rows as u64 + 4 * u64::from(history.page_rows)
            );
        }
        Charged {
            reservation: history.rows as u64,
            family,
            listed: listings.tiles.iter().map(|&tiles| class(tiles)).collect(),
            unlisted: class(0),
        }
    }

    /// An attention block's graph scratch follows neither the store's
    /// reservation of history rows (an address-space ceiling from the
    /// device's bytes) nor the context limit, and listing a launch's history
    /// row tiles adds nothing to it: on Metal over K8/V4 history every
    /// 512-row class that lists tiles is charged what the class that lists
    /// none is (the DIRECT and COISSUE forms decode into the rows of the
    /// partial outputs their key partitions leave), so the family holds the
    /// same bytes with and without tensor operations.
    #[test]
    fn attention_graph_scratch_follows_neither_the_reservation_nor_the_context() {
        for heads in [(4, 256), (1, 128), (2, 128), (1, 256), (2, 512), (4, 512)] {
            for context in [65_536, 262_144] {
                let plain = charged(heads, context, 8 << 30, false);
                let small = charged(heads, context, 8 << 30, true);
                let large = charged(heads, context, 256 << 30, true);
                assert!(large.reservation > small.reservation);
                assert!(!small.listed.is_empty());
                for &listed in small
                    .listed
                    .iter()
                    .chain(&large.listed)
                    .chain(&plain.listed)
                {
                    assert_eq!(listed, small.unlisted, "{heads:?} at {context}");
                }
                assert_eq!(
                    (small.family, small.unlisted),
                    (large.family, large.unlisted),
                    "{heads:?} at {context}: graph scratch follows the history reservation"
                );
                assert_eq!(
                    (plain.family, plain.unlisted),
                    (small.family, small.unlisted)
                );
                assert_eq!(
                    charged(heads, context, 256 << 30, false).family,
                    plain.family
                );
                eprintln!(
                    "kv heads and width {heads:?}, context {context}, reservation {} / {} rows: family {} bytes, 512-row attention class {} bytes in {} listing classes and the one listing none",
                    small.reservation,
                    large.reservation,
                    small.family,
                    small.unlisted,
                    small.listed.len()
                );
            }
        }
    }

    /// Every exact entry and block class seals into the layout certified for
    /// its structural regime and is charged that layout, and the family
    /// charges exactly the binding constants the exact classes bind.
    #[test]
    fn every_exact_target_class_seals_into_its_certified_layout() {
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 64,
            max_selected_rows: 64,
            max_drafting_slots: 64,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
        };
        for mut configuration in [QWEN35_CONFIGURATIONS[0], QWEN35_CONFIGURATIONS[3]] {
            // One recurrent and one attention block shape.
            configuration.blocks = configuration.attention_interval;
            let (definition, manifest) = declared_model(&configuration);
            for (backend, codec) in [
                BackendName::Cpu,
                BackendName::Metal,
                BackendName::Cuda,
                BackendName::Vulkan,
            ]
            .into_iter()
            .flat_map(|backend| [KvCodec::Dense, KvCodec::AffineK8V4].map(|codec| (backend, codec)))
            {
                let load = ModelLoadPlan::derive(
                    &manifest,
                    &definition,
                    ComponentSelection {
                        head: false,
                        vision: false,
                    },
                    resident_layout(ExecutionPath::Native, backend),
                )
                .unwrap();
                let state = ResourcePlanner::state_plan(
                    &definition,
                    &load,
                    PlannedMethod::Plain,
                    codec,
                    limits,
                    ResourceCapacity {
                        domain_bytes: 64 * 1024 * 1024 * 1024,
                        tensor_operations: crate::TensorOperations::Absent,
                    },
                )
                .unwrap();
                let plan = load.program_plan(&definition, codec).unwrap();
                let certificate = certify_target_family(
                    backend,
                    &load,
                    &definition.decoder,
                    &state,
                    plan.target(),
                    limits,
                )
                .unwrap();
                let geometry = &definition.decoder;
                let max_segments = state.target_state().span_limit().next_power_of_two() as u64;
                let weight = embedding_weight(&load).unwrap();
                let elements = [("EW", weight.resident), ("A", activation(geometry))];
                // The graphs a load seals and their plans' host heap, from
                // each exact class's own topology: blocks of one shape seal
                // one plan per class (no block here reads a layer scale).
                let (_, tapped) = block_taps(plan.target(), geometry).unwrap();
                let shapes = block_graph_shapes(&load, geometry, &state, &tapped).unwrap();
                let sealed = std::cell::Cell::new((0usize, HostHeap::default()));
                let seals = |heap: HostHeap| {
                    let (graphs, held) = sealed.get();
                    sealed.set((graphs + 1, held.plus(heap).unwrap()));
                };
                let owned = std::cell::Cell::new(HostHeap::default());
                let shared = FamilyConstants::default();
                let fits =
                    |graph: NativeGraphMetadata, layout: &NativeGraphLayout, class: String| {
                        // Every class of a regime seals a plan of the heap
                        // its layout plans.
                        assert_eq!(
                            graph.plan_heap(),
                            layout.plan_heap(),
                            "{} {backend:?} {class} plan heap",
                            configuration.model
                        );
                        let charged = graph.seal_with_layout(layout).unwrap_or_else(|error| {
                            panic!(
                            "{} {backend:?} {class} cannot seal into its certified layout: {error}",
                            configuration.model
                        )
                        });
                        assert_eq!(charged, layout.storage_bytes(), "{backend:?} {class}");
                    };
                let mut constants = CheckedGraphFamilyResources::new();
                for rows in magnitude_batching::row_classes(limits.max_launch_rows)
                    .into_iter()
                    .map(|rows| rows as u64)
                {
                    for source in [EntryTokens::Uploaded, EntryTokens::Selected] {
                        let graph = entry_graph_topology(
                            NativeGraphMetadata::new(backend),
                            &elements,
                            weight,
                            geometry,
                            rows,
                            source,
                        )
                        .unwrap()
                        .graph;
                        seals(graph.plan_heap());
                        fits(
                            graph,
                            &certificate.entries[&source],
                            format!("entry rows={rows}"),
                        );
                    }
                    for (index, block) in geometry.blocks.iter().enumerate() {
                        for (rows, segments, slots, listed) in block_classes(
                            paired_block(block).unwrap().mixer.kind(),
                            rows,
                            limits.max_launch_slots as u64,
                            max_segments,
                            &block_listings(
                                &state,
                                backend,
                                plan.target().blocks()[index],
                                index,
                                rows,
                            )
                            .unwrap(),
                        ) {
                            let (graph, mut block_constants, block_owned) =
                                checked_block_graph_draft(
                                    NativeGraphMetadata::new(backend),
                                    &load,
                                    geometry,
                                    &state,
                                    plan.target().blocks()[index],
                                    None,
                                    index,
                                    rows,
                                    segments,
                                    slots,
                                    listed,
                                )
                                .unwrap();
                            let certified =
                                &certificate.blocks[index][&(RowForm::of(rows), listed > 0)];
                            // Every class of a regime prepares a graph that
                            // owns the vectors its template plans.
                            assert_eq!(
                                block_owned, certified.owned,
                                "{} {backend:?} block {index} rows={rows} owned vectors",
                                configuration.model
                            );
                            if !(0..index).any(|first| shapes[first] == shapes[index]) {
                                seals(graph.plan_heap());
                                owned.set(owned.get().plus(block_owned).unwrap());
                            }
                            for constant in &mut block_constants {
                                shared.share(constant);
                            }
                            fits(
                                graph,
                                &certified.layout,
                                format!(
                                    "block {index} rows={rows} segments={segments} slots={slots}"
                                ),
                            );
                            constants.include(
                                NativeGraphStorageBytes {
                                    workspace: 0,
                                    output: 0,
                                    upload: 0,
                                },
                                block_constants,
                            );
                        }
                    }
                }
                assert_eq!(
                    certificate.resources.binding_constant_bytes,
                    constants.finish().unwrap().binding_constant_bytes,
                    "{} {backend:?} {codec:?} constants",
                    configuration.model
                );
                assert!(certificate.sealed.exact());
                let planned = certificate.sealed.at_most;
                assert!(planned.plans.bytes > 0 && planned.owned.bytes > 0);
                assert_eq!(
                    (planned.graphs, planned.plans),
                    sealed.get(),
                    "{} {backend:?} {codec:?} sealed graphs",
                    configuration.model
                );
                assert_eq!(
                    planned.owned,
                    owned.get(),
                    "{} {backend:?} {codec:?} owned vectors",
                    configuration.model
                );
                // The values the exact classes bind, each held once, are
                // the ones the plan counted.
                assert_eq!(
                    certificate.sealed.constants,
                    shared.host_heap().unwrap(),
                    "{} {backend:?} {codec:?} constant values",
                    configuration.model
                );
            }
        }
    }
}
