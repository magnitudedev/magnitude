//! Seismic-owned decoder block graphs. The engine supplies semantic model
//! topology and external state; checked entry contracts derive every stage
//! edge and intermediate allocation.

use crate::{
    native::{AttestedTarget, AttestedTargetBlock, OutputScales},
    operators::block::{
        self, BlockControlPorts, BlockStatePorts, FeedForwardEntries, MixerEntries, PerLayerParts,
    },
    operators::per_layer::graph::{
        per_layer_entry, per_layer_entry_class_slice, CheckedPerLayerEntryEntries,
        PerLayerEntryEntries, PerLayerEntryPorts,
    },
    programs::graph::draft::GraphDraft,
    programs::graph::tap::{TapEntry, TapPorts, TapPositions, Taps},
    programs::graph::GraphError,
    programs::graph::RowForm,
    programs::native_constants::{
        distinct_storage_bytes, CheckedGraphFamilyResources, CheckedGraphResources,
        ConstantTensors, GraphConstant,
    },
    ModelLoadPlan, ResidentTarget, ResourceLimits, StateResourcePlan, TargetBlockProgramSlot,
    TargetProgramPlan,
};
use crate::host_tables::HostTable;
use crate::operators::{self, paired_block, MixerKind, PairedBlock};
use magnitude_family_contracts::{
    Decoder, EmbeddingScale, SublayerIndex, TapPoint, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::embedding_rows;
use magnitude_state::LayerRef;
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, NativeGraphClassSlice, NativeGraphFamily,
    NativeGraphLayout, NativeGraphMetadata, NativeGraphPlan, NativeGraphStorageBytes, NativePort,
    WorkflowTensor,
};
use std::collections::BTreeMap;
use std::time::Instant;

/// A block graph's class: (rows, history segments, request slots).
type BlockClass = (u64, u64, u64);

/// The classes a block's sealed graph distinguishes for one row class. An
/// attention block reads history segments and is independent of request
/// slots; a recurrent block binds exact per-slot bank tables (every slot
/// count a launch of `rows` rows can serve, up to the launch slot bound) and
/// is independent of segments. Everything else is per row.
fn block_classes(
    mixer: MixerKind,
    rows: u64,
    max_slots: u64,
    max_segments: u64,
) -> Vec<BlockClass> {
    match mixer {
        MixerKind::Attention => std::iter::successors(Some(1u64), |segments| {
            segments
                .checked_mul(2)
                .filter(|segments| *segments <= max_segments)
        })
        .map(|segments| (rows, segments, 1))
        .collect(),
        MixerKind::Recurrent => (1..=rows.min(max_slots))
            .map(|slots| (rows, 1, slots))
            .collect(),
    }
}

/// The class of a block's graph serving a launch of `rows` rows over
/// `segments` history segments and `slots` requests.
fn block_class(mixer: MixerKind, rows: u64, segments: u64, slots: u64) -> BlockClass {
    match mixer {
        MixerKind::Attention => (rows, segments, 1),
        MixerKind::Recurrent => (rows, 1, slots),
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
) -> Result<NativeGraphClassSlice, String> {
    let mixer = block.mixer.kind();
    let classes = block_classes(mixer, rows, max_slots, max_segments);
    let slice = NativeGraphClassSlice::new()
        .dimension("M", [rows])
        .dimension("O", [rows]);
    let slice = match mixer {
        MixerKind::Attention => {
            slice.dimension("R", classes.iter().map(|&(_, segments, _)| segments))
        }
        MixerKind::Recurrent => slice.dimension("B", classes.iter().map(|&(_, _, slots)| slots)),
    };
    block::feed_forward_class_slice(block, rows, slice)
}

#[derive(Clone)]
pub struct PreparedTargetGraphs {
    entries: BTreeMap<(u64, EntryTokens), PreparedTargetEntryGraph>,
    /// Per block, its graph for each class it distinguishes.
    blocks_by_class: Vec<BTreeMap<BlockClass, PreparedTargetBlockGraph>>,
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
/// sharing identical block plans, and wall-clock seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SealReport {
    pub classes: usize,
    pub sealed_graphs: usize,
    pub seconds: f64,
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
            TapPoint::Exit => &mut tapped.last_mut().ok_or("a tapped target has no block")?.output,
            TapPoint::Sublayer(index) => {
                return Err(format!("the target taps no residual entering {index:?}"))
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
        _ => return Err(format!("block graph weight {role:?} is not a target sublayer role")),
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
            .map_or(Ok(0), |(element, extents)| element.canonical_byte_len(&extents))
            .map_err(|error| error.to_string())?;
        let per_layer_rows = per_layer_rows_bytes(self.per_layer_rows)?;
        distinct_storage_bytes(
            self.blocks_by_class
                .iter()
                .flat_map(|graphs| graphs.values().flat_map(|graph| graph.constants.iter())),
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
    ) -> Result<Self, String> {
        let mixers = geometry
            .blocks
            .iter()
            .map(|block| {
                paired_block(block)
                    .map(|paired| paired.mixer.kind())
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
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
                .max_visible_spans()?
                .checked_next_power_of_two()
                .ok_or("target segment class overflows")?,
        )
        .map_err(|_| "target segment class exceeds u64")?;
        // Blocks whose sealed graph would be identical share one plan per
        // class: the plan names weights by kind only, and each block binds
        // its own resident weights to it.
        let (tap_width, tapped) = block_taps(plan, geometry)?;
        let shapes = block_graph_shapes(load, geometry, state, &tapped)?;
        let mut blocks_by_class: Vec<BTreeMap<BlockClass, PreparedTargetBlockGraph>> =
            vec![BTreeMap::new(); handles.blocks.len()];
        let mut launch_classes = std::collections::BTreeSet::new();
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
        for rows in row_classes.into_iter().map(|rows| rows as u64) {
            for source in [EntryTokens::Uploaded, EntryTokens::Selected] {
                let entry = PreparedTargetEntryGraph::prepare(
                    device,
                    &handles.embedding,
                    load,
                    geometry,
                    rows,
                    source,
                    &certificate.entries[&source],
                )?;
                sealed_graphs += 1;
                max_output_bytes = max_output_bytes.max(entry.plan.output_bytes());
                plans.push(entry.plan.clone());
                entries.insert((rows, source), entry);
            }
            if let Some(per_layer) = &per_layer {
                let entry = per_layer.prepare(device, rows)?;
                sealed_graphs += 1;
                max_output_bytes = max_output_bytes.max(entry.plan.output_bytes());
                plans.push(entry.plan.clone());
                per_layer_entries.insert(rows, entry);
            }
            for (index, handle) in handles.blocks.iter().enumerate() {
                let mixer = mixers[index];
                // Output scales are values of the sealed graph.
                let shared = (0..index).find(|&first| {
                    shapes[first] == shapes[index]
                        && handles.blocks[first].output_scales == handle.output_scales
                });
                for class @ (rows, segments, slots) in
                    block_classes(mixer, rows, max_slots, max_segments)
                {
                    launch_classes.insert(class);
                    if let Some(first) = shared {
                        let graph = blocks_by_class[first][&class].clone();
                        blocks_by_class[index].insert(class, graph);
                        continue;
                    }
                    let began = Instant::now();
                    let positions = tapped[index].positions();
                    let tap = match (positions.any(), tap_width, &handles.taps) {
                        (false, ..) => None,
                        (true, Some(width), Some(taps)) => Some(TapEntry {
                            entry: &taps.tap,
                            width,
                            positions,
                        }),
                        _ => return Err("a tapped block has no tap entry".into()),
                    };
                    let block = PreparedTargetBlockGraph::prepare(device, handle, tap,
                        plan.blocks()[index].per_layer(),
                        load, geometry, state, index, rows, segments, slots,
                        &certificate.blocks[index][&RowForm::of(rows)])
                        .map_err(|error| format!(
                            "target graph row class {rows}, history segments {segments}, request slots {slots}, block {index}: {error}"
                        ))?;
                    sealed_graphs += 1;
                    max_output_bytes = max_output_bytes.max(block.plan.output_bytes());
                    plans.push(block.plan.clone());
                    blocks_by_class[index].insert(class, block);
                    if trace {
                        eprintln!(
                            "target graph sealed rows={rows} segments={segments} slots={slots} block={index} elapsed_ms={:.3}",
                            began.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
            }
        }
        let seal_seconds = began_all.elapsed().as_secs_f64();
        let family = NativeGraphFamily::new(&plans).map_err(|error| error.to_string())?;
        let seal = SealReport {
            classes: launch_classes.len(),
            sealed_graphs,
            seconds: seal_seconds,
        };
        Ok(Self {
            entries,
            blocks_by_class,
            mixers,
            classes: launch_classes.len(),
            family,
            max_output_bytes,
            blocks: handles.blocks.len(),
            tap_buffer: tap_width.map(|width| (activation(geometry), [largest_rows, width])),
            taps: tapped,
            per_layer_entries,
            per_layer_rows: per_layer_rows(plan, largest_rows),
            seal,
        })
    }

    /// The tap indices block `index` writes.
    pub fn block_taps(&self, index: usize) -> BlockTaps {
        self.taps[index]
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

    pub(crate) fn bind_weights(
        &self,
        resident: &ResidentTarget,
    ) -> Result<BoundTargetGraphs, String> {
        let mut entry_bound = BTreeMap::new();
        for (key, entry) in &self.entries {
            let fixed = [(&entry.table, resident.embedding.tensor())];
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
            let mut block_bound = BTreeMap::new();
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
                block_bound.insert(
                    *class,
                    graph.plan.bind_static(&fixed).map_err(|error| {
                        format!(
                            "target graph static binding class {class:?} block {index}: {error}"
                        )
                    })?,
                );
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
                            &weights.projection.tensor().device(), Element::f32(), &[0], &[],
                        ).map_err(|error| error.to_string())?;
                        let fixed = [
                            (&graph.ports.projection, weights.projection.tensor()),
                            (&graph.ports.norm, weights.norm.tensor()),
                            (&graph.ports.absent_scale, &absent_scale),
                        ];
                        graph
                            .plan
                            .bind_static(&fixed)
                            .map(|bound| (*rows, bound))
                            .map_err(|error| format!("per-layer entry graph rows {rows}: {error}"))
                    })
                    .collect::<Result<_, String>>()?,
                table: weights.table.clone(),
                rows: seismic::Tensor::zeros(
                    &resident.embedding.tensor().device(),
                    Element::f32(),
                    &extents,
                )
                .map_err(|error| format!("per-layer rows allocation failed: {error}"))?,
            }),
            _ => return Err("resident per-layer weights disagree with the prepared graphs".into()),
        };
        Ok(BoundTargetGraphs {
            prepared: self.clone(),
            entry_bound,
            bound,
            constants: constants.into_tensors(),
            taps,
            per_layer,
        })
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
    pub prepared: PreparedTargetGraphs,
    pub entry_bound: BTreeMap<(u64, EntryTokens), BoundNativeGraphPlan>,
    /// Per block, its bound graph for each class it distinguishes.
    pub bound: Vec<BTreeMap<BlockClass, BoundNativeGraphPlan>>,
    constants: Vec<seismic::Tensor>,
    /// The draft input rows of a tapped target: tapped blocks write their
    /// column blocks of a step's leading rows, and the readout fuses them.
    pub taps: Option<seismic::Tensor>,
    pub per_layer: Option<BoundPerLayerEntry>,
}

/// The bound per-layer entry of a per-layer model.
pub(crate) struct BoundPerLayerEntry {
    /// The graph per row class, bound to its weights.
    bound: BTreeMap<u64, BoundNativeGraphPlan>,
    /// The host table whose rows each step gathers.
    pub table: HostTable,
    /// The per-layer rows: the entry writes a step's leading rows, and every
    /// per-layer input sublayer reads them.
    pub rows: seismic::Tensor,
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
        index: usize,
    ) -> Result<(&PreparedTargetBlockGraph, &BoundNativeGraphPlan), String> {
        let mixer = *self
            .prepared
            .mixers
            .get(index)
            .ok_or_else(|| format!("target block {index} does not exist"))?;
        let key = block_class(mixer, rows, segments, slots);
        let graph = self.prepared.blocks_by_class[index]
            .get(&key)
            .ok_or_else(|| format!("target graph class {key:?} block {index} was not sealed"))?;
        let bound = self.bound[index]
            .get(&key)
            .ok_or_else(|| format!("target graph class {key:?} block {index} was not bound"))?;
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
        let weight = embedding_weight(load)?;
        let (graph, table, tokens, hidden) = entry_graph_topology(
            device.native_graph_with_layout(layout),
            embedding,
            weight,
            geometry,
            rows,
            source,
        )
        .map_err(|error| error.to_string())?;
        let plan = GraphDraft::seal(graph).map_err(|error| error.to_string())?;
        Ok(Self {
            plan,
            table,
            tokens,
            hidden,
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

fn entry_graph_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    embedding: G::Binding<'a, embedding_rows::Entry>,
    weight: &crate::WeightPlan,
    geometry: &Decoder,
    rows: u64,
    source: EntryTokens,
) -> Result<(G, NativePort, NativePort, WorkflowTensor), GraphError> {
    let table = graph.port(weight.resident, &weight.shape)?;
    let dimensions = [
        ("M", rows),
        ("V", geometry.vocabulary),
        ("D", geometry.hidden),
    ];
    let tokens = match source {
        EntryTokens::Uploaded => graph.input_for(embedding, "tokens", &dimensions)?,
        EntryTokens::Selected => {
            graph.port_with_class_extent(Element::i32(), &[rows, 2], 0, "M")?
        }
    };
    let (scale, normalize, epsilon) = embedding_transform(geometry);
    let result = graph.enqueue::<embedding_rows::Entry>(
        embedding,
        &dimensions,
        embedding_rows::WorkflowArgs {
            table: table.tensor().into(),
            tokens: tokens.tensor().into(),
            scale,
            normalize,
            epsilon,
        },
    )?;
    graph.export(&result.r1)?;
    Ok((graph, table, tokens, result.r1))
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
    let (graph, _, _, _) = entry_graph_topology(
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
    GraphDraft::seal(graph)
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
}

impl WeightPart {
    /// This part of `weight`.
    pub(crate) fn of(self, weight: &crate::ResidentWeight) -> Result<&seismic::Tensor, String> {
        match self {
            Self::Values => Ok(weight.tensor()),
            Self::Scale => weight.scale().ok_or_else(|| {
                format!(
                    "resident weight {:?} has no second-level scale",
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
) -> Result<seismic::WorkflowTensor, GraphError> {
    let plan = load
        .weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| format!("missing planned weight {role:?}"))?;
    let port = match (part, &plan.scale) {
        (WeightPart::Values, _) => graph.port(plan.resident, &plan.shape),
        (WeightPart::Scale, Some(scale)) => graph.port(Element::f32(), &[scale.extent]),
        (WeightPart::Scale, None) => {
            return Err(format!("planned weight {role:?} has no second-level scale").into())
        }
    }?;
    let tensor = port.tensor().clone();
    ports.push((WeightPort { role, part }, port));
    Ok(tensor)
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
    weight_port(graph, load, WeightRole { scope, kind }, WeightPart::Values, ports)
}

/// A projection weight and the tensor its entry's accumulator-scale port
/// binds: its resident second-level scale, else an absent scale (zero
/// extent, never read), and that port's static extent.
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
    let role = WeightRole { scope, kind };
    let weight = weight_port(graph, load, role, WeightPart::Values, ports)?;
    let extent = planned_scale_extent(load, role)?;
    let scale = match extent {
        0 => GraphConstant::absent_scale(graph, constants)?,
        1 => weight_port(graph, load, role, WeightPart::Scale, ports)?,
        _ => return Err(format!("{role:?} has a per-matrix scale at a per-tensor port").into()),
    };
    Ok(ScaledWeight {
        weight,
        scale,
        extent,
    })
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
        _ => weight_port(graph, load, role, WeightPart::Scale, ports).map(Some),
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
        layout: &NativeGraphLayout,
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
                output_scales: handle.output_scales,
            },
            &mut weights,
            &mut constants,
        )
        .map_err(|error| error.to_string())?;
        graph.export(&parts.output).map_err(|error| error.to_string())?;
        let plan = graph.seal().map_err(|error| error.to_string())?;
        Ok(Self {
            plan,
            hidden,
            weights,
            constants,
            state: parts.state,
            controls: parts.controls,
            taps: parts.taps,
            per_layer: parts.per_layer,
            output: parts.output,
        })
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
        output_scales,
    } = inputs;
    let block = geometry
        .blocks
        .get(block_index)
        .ok_or("target block geometry is absent")?;
    let paired = paired_block(block).map_err(|error| error.to_string())?;
    let sublayers = block::BlockSublayers {
        paired: &paired,
        load,
        geometry,
        state,
        block_index,
        rows,
        segments,
        slots,
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
    let output =
        sublayers.feed_forward(graph, feed_forward_entries, mixed, weights, constants)?;
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
/// without forming kernels or allocating device storage.
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
) -> Result<(NativeGraphMetadata, Vec<GraphConstant>), GraphError> {
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
    let checked_mixer = block::CheckedMixerEntries::new(slot.mixer());
    let mixer = checked_mixer.entries()?;
    let checked_feed_forward = slot.feed_forward().map(block::CheckedFeedForwardEntries::new);
    let feed_forward = checked_feed_forward.as_ref().map(|checked| checked.entries());
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
    Ok((graph, constants))
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
) -> Result<(NativeGraphStorageBytes, Vec<GraphConstant>), GraphError> {
    let (graph, constants) = checked_block_graph_draft(
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
    )?;
    Ok((GraphDraft::seal(graph)?, constants))
}

/// The same decoder class ladder as `PreparedTargetGraphs::prepare`. The
/// family takes independent maxima because its workspace, output and upload
/// holdings may peak in different members.
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

struct TargetFamilyCertificate {
    resources: CheckedGraphResources,
    entries: BTreeMap<EntryTokens, NativeGraphLayout>,
    /// The per-layer entry graph's layout, of a per-layer entry.
    per_layer: Option<NativeGraphLayout>,
    blocks: Vec<BTreeMap<RowForm, NativeGraphLayout>>,
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
            .max_visible_spans()?
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
        let (graph, _, _, _) = entry_graph_topology(
            NativeGraphMetadata::new_template(backend),
            &elements,
            weight,
            geometry,
            largest_rows,
            source,
        )?;
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
            let (graph, constants) = checked_block_graph_draft(
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
            )?;
            let slices = rows
                .iter()
                .map(|&rows| block_class_slice(&paired, rows, max_slots, max_segments))
                .collect::<Result<Vec<_>, _>>()?;
            let layout = graph
                .seal_template()
                .and_then(|template| template.certify(&slices))
                .map_err(|error| format!("target block {index}: {error}"))?;
            family.include(layout.storage_bytes(), constants);
            blocks[index].insert(form, layout);
        }
        // Row-class constants (`block::class_constants_of`).
        for &rows in &row_classes {
            let constants = block::class_constants_of(&paired, geometry.hidden, rows)?;
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
mod resource_template_tests {
    use super::*;
    use crate::{
        assessment::plan::tests::{declared_model, QWEN35_CONFIGURATIONS},
        resident_layout, ComponentSelection, ExecutionPath, PlannedMethod, ResourceCapacity,
        ResourcePlanner,
    };
    use magnitude_state::KvCodec;

    /// Every exact entry and block class seals into the layout certified for
    /// its structural regime and is charged that layout, and the family
    /// charges exactly the binding constants the exact classes bind.
    #[test]
    fn every_exact_target_class_seals_into_its_certified_layout() {
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 64,
            max_projected_rows: 64,
            max_images_per_request: 1,
            lookahead: false,
        };
        for mut configuration in [QWEN35_CONFIGURATIONS[0], QWEN35_CONFIGURATIONS[3]] {
            // One recurrent and one attention block shape.
            configuration.blocks = configuration.attention_interval;
            let (definition, manifest) = declared_model(&configuration);
            for backend in [
                BackendName::Cpu,
                BackendName::Metal,
                BackendName::Cuda,
                BackendName::Vulkan,
            ] {
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
                    KvCodec::Dense,
                    limits,
                    ResourceCapacity {
                        domain_bytes: 64 * 1024 * 1024 * 1024,
                    },
                )
                .unwrap();
                let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
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
                let max_segments = state
                    .target_state()
                    .max_visible_spans()
                    .unwrap()
                    .next_power_of_two() as u64;
                let weight = embedding_weight(&load).unwrap();
                let elements = [("EW", weight.resident), ("A", activation(geometry))];
                let fits =
                    |graph: NativeGraphMetadata, layout: &NativeGraphLayout, class: String| {
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
                        let (graph, _, _, _) = entry_graph_topology(
                            NativeGraphMetadata::new(backend),
                            &elements,
                            weight,
                            geometry,
                            rows,
                            source,
                        )
                        .unwrap();
                        fits(
                            graph,
                            &certificate.entries[&source],
                            format!("entry rows={rows}"),
                        );
                    }
                    for (index, block) in geometry.blocks.iter().enumerate() {
                        for (rows, segments, slots) in block_classes(
                            paired_block(block).unwrap().mixer.kind(),
                            rows,
                            limits.max_launch_slots as u64,
                            max_segments,
                        ) {
                            let (graph, block_constants) = checked_block_graph_draft(
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
                            )
                            .unwrap();
                            fits(
                                graph,
                                &certificate.blocks[index][&RowForm::of(rows)],
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
                    "{} {backend:?} constants",
                    configuration.model
                );
            }
        }
    }
}
