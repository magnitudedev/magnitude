//! The grouped form's grouping and expert reads, which the fused and the
//! general routed forms share (`fused_graph`, `graph`): `routed_group` over
//! a graph's rows or a merge set's, and where the experts it groups are
//! read. Each form's expert entry (`routed_experts`, or the general form's
//! gated or up-only one) runs over what the grouping wrote.

use super::fused_graph::{TILE_ROWS, absent_map, class_rows, grouped_blocks};
use super::split::PassPorts;
use super::streamed::StreamedLayer;
use crate::planning::RecordTensors;
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use magnitude_kernels::routed_group;
use seismic::{NativePort, WorkflowTensor};

/// The grouped experts' rows over the blocks the grouping filled, each
/// grouped choice's position in them (`inverse`), and the blocks they
/// hold.
pub(crate) struct ExpertsPass {
    pub output: WorkflowTensor,
    pub inverse: WorkflowTensor,
    pub blocks: u64,
}

/// Where the grouped experts are read, each form's expert stacks `S`:
/// expert `e` from `stacks` (the resident stacks, or a streamed layer's
/// scratch tier), or, through a layer's map at the step (the first two),
/// from its group's records (the third, of the fourth's count) wherever the
/// map holds it for the step.
struct Sources<S> {
    stacks: S,
    map: Option<(WorkflowTensor, WorkflowTensor, S, u64)>,
}

/// What an expert entry reads beside the grouping: the stacks by expert,
/// the map (`[X, MAP_WORDS]`), its rows `mapped` (`X`, 0 without a map),
/// and the step it holds records for (`[QS]`, none without a map), and the
/// records it places experts in and their count (`R`; the stacks, of the
/// experts, without a map).
pub(super) struct Reads<S> {
    pub stacks: S,
    pub map: WorkflowTensor,
    pub step: WorkflowTensor,
    pub records: S,
    pub count: u64,
    pub mapped: u64,
}

impl<S> Reads<S> {
    /// `QS`, the step's words: a step pair where the reads go through a
    /// map, none without one.
    pub fn step_words(&self) -> u64 {
        match self.mapped {
            0 => 0,
            _ => magnitude_kernels::constants::STEP_WORDS,
        }
    }
}

/// What a form's expert entry runs over: `rows_read[0]` rows from row
/// `first[0]` of `rows` rows of the experts' `input`, grouped into `blocks`
/// tiles (the grouping's `order`, `block_experts` and `used`), read through
/// `reads`.
pub(super) struct Grouped<'t, S> {
    pub rows: u64,
    pub blocks: u64,
    pub first: &'t WorkflowTensor,
    pub rows_read: &'t WorkflowTensor,
    pub input: &'t WorkflowTensor,
    pub order: &'t WorkflowTensor,
    pub block_experts: &'t WorkflowTensor,
    pub used: &'t WorkflowTensor,
    pub reads: Reads<S>,
}

/// A whole layer's grouped experts over its class's `rows` rows of
/// `[routes, input]`, every row from row 0, each read from `stacks` by
/// expert, of `[experts, selected]`; the grouping's tables are graph
/// locals. `expand` runs the form's expert entry.
#[allow(clippy::too_many_arguments)]
pub(super) fn class<'a, G: GraphDraft + 'a, S: Clone>(
    graph: &mut G,
    group: G::Binding<'a, routed_group::Entry>,
    [routes, input]: [&WorkflowTensor; 2],
    stacks: S,
    rows: u64,
    [experts, selected]: [u64; 2],
    constants: &mut Vec<GraphConstant>,
    expand: impl FnOnce(&mut G, Grouped<'_, S>) -> Result<WorkflowTensor, GraphError>,
) -> Result<ExpertsPass, GraphError> {
    let blocks = grouped_blocks(rows, experts, selected)?;
    let [first, rows_read] = class_rows(graph, rows, constants)?;
    grouped(
        graph,
        group,
        [routes, input],
        [&first, &rows_read],
        [None, None],
        Sources { stacks, map: None },
        rows,
        blocks,
        [experts, selected],
        constants,
        expand,
    )
}

/// A merge set's pass (`OnceStage::Pass`) over a layer of `[experts,
/// selected]`: the crossed rows its `ports` name of the pass's `rows`
/// grouped into its merged positions, then the experts read where they
/// live: a streamed `layer`'s in its group's records (`stacks`, its weight
/// ports), wherever its slot map places them, else its scratch tier (the
/// form's `scratch` view, slot `e` its expert `e`), counting each expert's
/// choices into its group's reads; a resident one's `stacks`. `expand` runs
/// the form's expert entry, whose rows the pass stores into its merged
/// experts.
#[allow(clippy::too_many_arguments)]
pub(super) fn pass<'a, G: GraphDraft + 'a, S: Clone>(
    graph: &mut G,
    group: G::Binding<'a, routed_group::Entry>,
    layer: Option<&mut StreamedLayer<'_, 'a, G>>,
    ports: &mut PassPorts,
    stacks: S,
    scratch: impl FnOnce(&RecordTensors<NativePort>) -> Result<S, GraphError>,
    rows: u64,
    [experts, selected]: [u64; 2],
    constants: &mut Vec<GraphConstant>,
    expand: impl FnOnce(&mut G, Grouped<'_, S>) -> Result<WorkflowTensor, GraphError>,
) -> Result<WorkflowTensor, GraphError> {
    let mut placed = layer.map(|layer| layer.pass_sources(graph)).transpose()?;
    let (sources, reads) = match &mut placed {
        Some((map, step, tier, records, reads)) => (
            Sources {
                stacks: scratch(tier)?,
                map: Some((map.clone(), step.clone(), stacks, *records)),
            },
            Some(reads),
        ),
        None => (Sources { stacks, map: None }, None),
    };
    let PassPorts { crossing, experts: into, inverse, rows: read, first, blocks } = ports;
    let pass = grouped(
        graph,
        group,
        [crossing.routes.tensor(), crossing.normalized.tensor()],
        [first.tensor(), read.tensor()],
        [Some(inverse), reads],
        sources,
        rows,
        *blocks,
        [experts, selected],
        constants,
        expand,
    )?;
    graph.store(&pass.output, into)?;
    Ok(pass.output)
}

/// The `rows_read[0]` rows from row `first[0]` of `rows` rows of
/// `[routes, input]` grouped by expert into `blocks` tiles (each choice's
/// position into `inverse`, and each expert's choices into `counts`, graph
/// locals where none is given), then `expand` over the blocks the grouping
/// filled, reading `sources`.
#[allow(clippy::too_many_arguments)]
fn grouped<'a, G: GraphDraft + 'a, S: Clone>(
    graph: &mut G,
    group: G::Binding<'a, routed_group::Entry>,
    [routes, input]: [&WorkflowTensor; 2],
    [first, rows_read]: [&WorkflowTensor; 2],
    [inverse, counts]: [Option<&mut NativePort>; 2],
    sources: Sources<S>,
    rows: u64,
    blocks: u64,
    [experts, selected]: [u64; 2],
    constants: &mut Vec<GraphConstant>,
    expand: impl FnOnce(&mut G, Grouped<'_, S>) -> Result<WorkflowTensor, GraphError>,
) -> Result<ExpertsPass, GraphError> {
    let tables = [
        ("M", rows),
        ("E", experts),
        ("K", selected),
        ("B", blocks),
        ("T", TILE_ROWS),
    ];
    let mut order = graph.local_for(group, "order", &tables)?;
    let mut local = None;
    let inverse = match inverse {
        Some(inverse) => inverse,
        None => local.insert(graph.local_for(group, "inverse", &tables)?),
    };
    let mut local_counts = None;
    let counts = match counts {
        Some(counts) => counts,
        None => local_counts.insert(graph.local_for(group, "counts", &tables)?),
    };
    let mut block_experts = graph.local_for(group, "blocks", &tables)?;
    let mut used = graph.local_for(group, "used", &tables)?;
    graph.enqueue(
        group,
        &tables,
        routed_group::WorkflowArgs {
            first: first.into(),
            rows: rows_read.into(),
            routes: routes.into(),
            counts: counts.tensor_mut().into(),
            order: order.tensor_mut().into(),
            inverse: inverse.tensor_mut().into(),
            blocks: block_experts.tensor_mut().into(),
            used: used.tensor_mut().into(),
        },
    )?;
    let reads = match sources.map {
        Some((map, step, records, count)) => Reads { stacks: sources.stacks, map, step, records, count, mapped: experts },
        None => Reads {
            records: sources.stacks.clone(),
            stacks: sources.stacks,
            map: absent_map(graph, constants)?,
            step: GraphConstant::empty_of(graph, seismic::Element::i32(), &[0], constants)?,
            count: experts,
            mapped: 0,
        },
    };
    let output = expand(
        graph,
        Grouped {
            rows,
            blocks,
            first,
            rows_read,
            input,
            order: order.tensor(),
            block_experts: block_experts.tensor(),
            used: used.tensor(),
            reads,
        },
    )?;
    Ok(ExpertsPass { output, inverse: inverse.tensor().clone(), blocks })
}
