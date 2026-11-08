use super::{
    GroupExtents, GroupRows,
    block::{BlockGraphInputs, BlockGraphs, SealedBlockGraph},
    bound::{
        BlockBindings, BoundPerLayerEntry, BoundTargetGraphs, GroupBuffers, PassBuffers, TokenRows,
        bind_block_graphs, zeros,
    },
    certify::{TargetFamilyCertificate, certify_target_family},
    classes::{
        BlockGraphEnumeration, BlockListings, BlockOutputScales, BlockServed, LaunchControlLayout,
        Passes, StagedClass, block_sealed_stages, block_stages, chained_once_runs, class_topology, decode_launch, enumerate_block_graphs, LaunchRuns, once_class,
        slot_selection,
    },
    entry::{
        EntryTokens, PerLayerEntryGraphSource, PreparedPerLayerEntryGraph,
        PreparedTargetEntryGraph, notice_bytes, notice_rows, notifies_selection,
    },
    heap::{GraphCounts, PlannedSealedGraphs, SealedGraphsHeap, SealedHeap, block_graph_storage},
};
use crate::{
    ModelLoadPlan, ResidentTarget, ResourceLimits, StateResourcePlan, TargetProgramPlan,
    native::AttestedTarget,
    operators,
    operators::{
        MixerKind, paired_block,
        routed::split::{OnceStage, RunStage, Stage},
        streams::StreamsGeometry,
    },
    programs::{
        graph::tap::{TapEntry, TapPositions},
        native_constants::{ConstantTensors, FamilyConstants, distinct_storage_bytes},
    },
    stored_table::SelectionNotice,
    streamed_table::StreamedTable,
};
use magnitude_family_contracts::{
    Decoder, ExitNorm, SublayerIndex, TapPoint, WeightKind, WeightScope,
};
use magnitude_kernels::selection_notice;
use magnitude_state::LayerRef;
use seismic::{Device, Element, HostHeap, NativeGraphFamily};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::Arc, time::Instant};

#[derive(Clone)]
pub struct PreparedTargetGraphs {
    pub(super) entries: BTreeMap<(u64, EntryTokens), PreparedTargetEntryGraph>,
    /// Per block, its graphs (`BlockGraphs`).
    pub(super) blocks_by_class: Vec<BlockGraphs>,
    /// Per block, its prefill rows' listing classes (`block_listings`).
    listings: Vec<BlockListings>,
    /// The controls of a launch's one upload of them.
    controls: LaunchControlLayout,
    /// The upload regions of a workspace slot, as the plan certified them
    /// (`TargetFamilyCertificate::upload_regions`).
    upload_regions: Vec<u64>,
    /// The most history segments a launch has.
    max_segments: u64,
    /// Block mixers, which decide the class a launch selects per block.
    pub(super) mixers: Vec<MixerKind>,
    /// Per block, its selected read's shape, which decides whether a launch
    /// takes its scoring class.
    selections: Vec<Option<crate::SelectionShape>>,
    family: NativeGraphFamily,
    max_output_bytes: u64,
    /// Per block, the tap indices its graph writes.
    taps: Vec<BlockTaps>,
    /// The first block of the stateless tail (`stateless_tail`).
    stateless_tail: usize,
    /// The per-layer entry graph per row class, of a per-layer entry.
    pub(super) per_layer_entries: BTreeMap<u64, PreparedPerLayerEntryGraph>,
    /// The rows of each copy of the selection notice a load with stored
    /// tables binds (`notice_rows`).
    notice_rows: Option<u64>,
    /// What its prompt groups' buffers span (`GroupRows`).
    group: GroupRows,
    /// How each block is served, which decides its stages, with its routed
    /// layer's values (`block_served`), whose crossing and merged positions
    /// a group holds over its rows (`group_split`).
    pub(super) served: Vec<BlockServed>,
    /// The passes a group runs (`group_pass_rows`).
    pub(super) passes: Passes,
    seal: SealReport,
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
#[derive(Clone, Debug, PartialEq)]
pub(super) struct BlockGraphShape {
    geometry: String,
    /// (sublayer, branch) of each role, its resident element, shape and
    /// accumulator-scale port extent.
    weights: Vec<((u32, Option<u32>), WeightKind, Element, Vec<u64>, u64)>,
    state: String,
    /// Decoder-boundary work belongs to the graph as well as its sublayers.
    /// Interior blocks cannot stand in for the stream widening or exit mix.
    widens_streams: bool,
    mixes_exit: bool,
    /// Where a separate draft taps the block.
    tapped: TapPositions,
    /// Whether the block exports its output residual to a head
    /// (`FamilyOutline::exports_residual`).
    residual: bool,
    /// Whether the block's routed layer is the load's first streamed one,
    /// which sets a bank-served launch's step.
    sets_step: bool,
    /// The layers a streamed routed layer's route predicts (its lead): each
    /// one's block offset from this block and the weights the route reads
    /// of it (`BankLayer::hinted`). A graph shared by blocks binds each
    /// one's layers at the same offsets (`block_scope`).
    predicts: Vec<(u32, Vec<crate::planning::HintWeight>)>,
    /// A streamed routed layer's bank binding.
    banked: Option<crate::BankBinding>,
    /// The block offset back to the streamed layer whose route predicts
    /// this block's (whose hint ring its route reads, `StreamedLayer::hinted`);
    /// none for a block that streams none or the first streamed one. A
    /// graph shared by blocks binds each one's predictor at the same offset.
    hinted_by: Option<u32>,
}

impl BlockGraphShape {
    /// The shape without its drafter's writes: no taps, no exported
    /// residual (`FamilyOutline::undrafted`).
    pub(super) fn undrafted(&self) -> Self {
        Self { tapped: TapPositions::default(), residual: false, ..self.clone() }
    }
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
pub(super) fn trailing_run(blocks: usize, stateless: impl Fn(usize) -> bool) -> usize {
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

/// A target sublayer (or branch) scope of a graph prepared for block `from`,
/// moved to block `to`: the same sublayer (or branch) at the same offset
/// from the block, so a scope of a later block (a streamed route's layers
/// ahead) moves with it. A decoder-wide scope (the exit site the last block
/// ends with) is every block's own.
pub(super) fn block_scope(scope: WeightScope, from: usize, to: usize) -> Result<WeightScope, String> {
    let moved = |block: u32| {
        (u64::from(block) + to as u64)
            .checked_sub(from as u64)
            .and_then(|block| u32::try_from(block).ok())
            .ok_or_else(|| format!("block graph scope {scope:?} moves outside the target"))
    };
    Ok(match scope {
        WeightScope::Target => WeightScope::Target,
        WeightScope::TargetSublayer(SublayerIndex { block, sublayer }) => {
            WeightScope::TargetSublayer(SublayerIndex { block: moved(block)?, sublayer })
        }
        WeightScope::TargetBranch {
            sublayer: SublayerIndex { block, sublayer },
            branch,
        } => WeightScope::TargetBranch {
            sublayer: SublayerIndex { block: moved(block)?, sublayer },
            branch,
        },
        _ => {
            return Err(format!(
                "block graph scope {scope:?} is not a target sublayer scope"
            ));
        }
    })
}

/// The layers block `index`'s streamed routed layer predicts
/// (`BlockGraphShape::predicts`); none for a block that streams none.
fn predicted_layers(
    load: &ModelLoadPlan,
    index: usize,
) -> Result<Vec<(u32, Vec<crate::planning::HintWeight>)>, String> {
    let Some(bank) = load.expert_bank() else {
        return Ok(Vec::new());
    };
    let Some((layer, _)) = bank.feed_forward(block_scopes(index)?[1]) else {
        return Ok(Vec::new());
    };
    bank.predicted(layer)
        .map(|predicted| {
            let predicted = &bank.layers()[predicted];
            let (WeightScope::TargetSublayer(SublayerIndex { block, .. })
            | WeightScope::TargetBranch {
                sublayer: SublayerIndex { block, .. },
                ..
            }) = predicted.scope
            else {
                return Err(format!("a streamed layer {:?} is not a target sublayer", predicted.scope));
            };
            let offset = (block as usize)
                .checked_sub(index)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or("a predicted layer precedes its block")?;
            Ok((offset, predicted.hinted.clone()))
        })
        .collect()
}

/// [`BlockGraphShape::hinted_by`] of block `index`.
fn hinted_by(load: &ModelLoadPlan, index: usize) -> Result<Option<u32>, String> {
    let Some(bank) = load.expert_bank() else {
        return Ok(None);
    };
    let Some((layer, _)) = bank.feed_forward(block_scopes(index)?[1]) else {
        return Ok(None);
    };
    if layer == 0 || bank.ahead(layer - 1) == 0 {
        return Ok(None);
    }
    let (WeightScope::TargetSublayer(SublayerIndex { block, .. })
    | WeightScope::TargetBranch { sublayer: SublayerIndex { block, .. }, .. }) = bank.layers()[layer - 1].scope
    else {
        return Err(format!("a streamed layer {:?} is not a target sublayer", bank.layers()[layer - 1].scope));
    };
    index
        .checked_sub(block as usize)
        .and_then(|offset| u32::try_from(offset).ok())
        .map(Some)
        .ok_or_else(|| "a predicting layer follows its block".into())
}

/// [`BlockGraphShape::banked`] of block `index`.
fn banked_layer(load: &ModelLoadPlan, index: usize) -> Result<Option<crate::BankBinding>, String> {
    let scope = block_scopes(index)?[1];
    load.expert_bank()
        .and_then(|bank| Some(bank.binding(bank.feed_forward(scope)?.0)))
        .transpose()
}

pub(super) fn block_graph_shapes(
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    tapped: &[BlockTaps],
    residual: Option<usize>,
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
                    let count = paired.bank_components();
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
                widens_streams: index == 0
                    && StreamsGeometry::of(geometry)
                        .map_err(|error| error.to_string())?
                        .is_some(),
                mixes_exit: index + 1 == geometry.blocks.len()
                    && matches!(geometry.exit.norm, ExitNorm::HyperConnectionHead(_)),
                tapped: tapped[index].positions(),
                residual: residual == Some(index),
                sets_step: {
                    let scope = block_scopes(index)?[1];
                    load.expert_bank().is_some_and(|bank| {
                        bank.feed_forward(scope).is_some_and(|(layer, _)| bank.sets_step(layer))
                    })
                },
                predicts: predicted_layers(load, index)?,
                banked: banked_layer(load, index)?,
                hinted_by: hinted_by(load, index)?,
            })
        })
        .collect()
}

impl PreparedTargetGraphs {
    /// Bytes binding allocates: the distinct graph constants, the floor
    /// group's buffers and the pass's (`GroupExtents`), and the selection
    /// notice (`notice_bytes`).
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        let floor = self.group_extents(1);
        let (group, pass, notice) = (floor.bytes()?, floor.pass_bytes()?, notice_bytes(self.notice_rows)?);
        distinct_storage_bytes(self.blocks_by_class.iter().flat_map(|graphs| {
            let runs = graphs.runs.iter().flat_map(|(_, graph)| graph.constants.iter());
            runs.chain(graphs.passes.iter().flat_map(|(_, graph)| graph.constants.iter()))
        }))?
        .checked_add(group)
        .and_then(|bytes| bytes.checked_add(pass))
        .and_then(|bytes| bytes.checked_add(notice))
        .ok_or_else(|| "target binding charge overflows".into())
    }

    /// The buffers a prompt group of `places` places holds.
    pub(super) fn group_extents(&self, places: usize) -> GroupExtents {
        self.group.extents(&self.served, places)
    }

    /// The bytes a prompt group of `places` places' own buffers hold beside
    /// the floor's (`GroupRows::group_bytes`).
    pub(crate) fn group_bytes(&self, places: usize) -> Result<u64, String> {
        self.group.group_bytes(&self.served, places)
    }

    /// `notice`: the selection notice of a load that has one; `landing`: the
    /// landing entries of the target's token table, when it has landing
    /// slots.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        device: &Device,
        handles: &AttestedTarget,
        notice: Option<&seismic::NativeKernel<selection_notice::Entry>>,
        landing: Option<&crate::native::AttestedLanding>,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        pass: crate::PassLaunches,
        workers: &seismic::Workers,
    ) -> Result<Self, String> {
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
            certify_target_family(device.backend(), load, geometry, state, plan, limits, pass)
                .map_err(|error| error.to_string())?;
        let TargetFamilyCertificate { outline, served, passes, group: certificate_group, .. } = certificate;
        let largest_rows = outline.largest_rows;
        let tap_width = outline.tap_width;
        let stateless_tail = stateless_tail(state, &outline.mixers, &outline.tapped);
        // Output scales are values of the sealed graph.
        let scales = handles
            .blocks
            .iter()
            .map(|handle| handle.output_scales)
            .collect::<Vec<_>>();
        // The (block, class) plans to seal, and those an earlier block of
        // the same shape seals (with the block that does), as enumerated.
        // Blocks whose sealed graph would be identical share one plan per
        // class: the plan names weights by kind only, and each block binds
        // its own resident weights to it.
        let BlockGraphEnumeration {
            sealing,
            sharing,
            listings,
            controls,
            launch_classes,
            exact: _,
        } = enumerate_block_graphs(
            &outline,
            &served,
            passes,
            state,
            plan,
            BlockOutputScales::Read(&scales),
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
        let notice = match notifies_selection(load) {
            true => Some(notice.ok_or("a stored table without the selection notice")?),
            false => None,
        };
        for &rows in &outline.row_classes {
            for source in [EntryTokens::Uploaded, EntryTokens::Selected] {
                let layout = &certificate.entries[&source];
                let entry = PreparedTargetEntryGraph::prepare(
                    device,
                    &handles.embedding,
                    notice,
                    landing,
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
        let seal_block = |&(index, staged): &(usize, StagedClass)| {
            let ((rows, segments, slots, listed, scored), stage) = staged;
            let began = Instant::now();
            let positions = outline.tapped[index].positions();
            let tap = match (positions.any(), tap_width, &handles.taps) {
                (false, ..) => None,
                (true, Some(width), Some(taps)) => Some(TapEntry {
                    entry: &taps.tap,
                    width,
                    positions,
                }),
                _ => return Err("a tapped block has no tap entry".to_owned()),
            };
            // A pass merges its own rows (`Passes::lengths`); a back combines
            // from the most any pass merges.
            let capacity = if stage == Stage::Once(OnceStage::Pass) { rows } else { passes.rows() };
            let block = SealedBlockGraph::prepare(
                device,
                &handles.blocks[index],
                tap,
                plan.blocks()[index],
                BlockGraphInputs {
                    output_scales: handles.blocks[index].output_scales,
                    residual: outline.exports_residual(index),
                    ..BlockGraphInputs::of_class(load, geometry, state, index, staged, capacity)
                },
                &certificate.blocks[index][&class_topology(staged)].layout,
                &constants,
            )
            .map_err(|error| {
                format!(
                    "target graph row class {rows}, history segments {segments}, request slots {slots}, listed {listed}, scored {scored}, {stage:?} of block {index}: {error}"
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
        // Each block's graphs go into vectors of exactly its classes of each
        // kind, in the order enumerated, which ascends by class.
        let mut class_counts = vec![GraphCounts::default(); handles.blocks.len()];
        let mut sealed_counts = GraphCounts::default();
        for &(index, (_, stage)) in &sealing {
            class_counts[index].count(stage);
            sealed_counts.count(stage);
        }
        for &(index, (_, stage), _) in &sharing {
            class_counts[index].count(stage);
        }
        let mut blocks_by_class = class_counts
            .iter()
            .map(|counts| BlockGraphs {
                runs: Vec::with_capacity(counts.runs),
                passes: Vec::with_capacity(counts.passes),
            })
            .collect::<Vec<_>>();
        for (&(index, class), block) in sealing.iter().zip(sealed) {
            let block = block?;
            sealed_graphs += 1;
            let certified = &certificate.blocks[index][&class_topology(class)];
            let graph = || format!("target graph class {class:?} of block {index}");
            let graphs = &mut blocks_by_class[index];
            let plan = match block {
                SealedBlockGraph::Run(block) => {
                    held.own(&block, certified, graph)?;
                    let plan = block.plan.clone();
                    graphs.runs.push((class, block));
                    plan
                }
                SealedBlockGraph::Pass(block) => {
                    held.own(&block, certified, graph)?;
                    let plan = block.plan.clone();
                    graphs.passes.push((class.0.0, block));
                    plan
                }
            };
            held.seal(&plan, &certified.layout, graph)?;
            max_output_bytes = max_output_bytes.max(plan.output_bytes());
            plans.push(plan);
        }
        // A block whose sealed graph is another's holds that block's
        // prepared graph: the one plan, ports and constants.
        for (index, class, first) in sharing {
            let shared = |sealed: Option<usize>| {
                sealed.ok_or_else(|| format!("block {first} sealed no graph of class {class:?}"))
            };
            match class.1 {
                Stage::Once(OnceStage::Pass) => {
                    let graphs = &blocks_by_class[first].passes;
                    let at = shared(graphs.binary_search_by(|(rows, _)| rows.cmp(&class.0.0)).ok())?;
                    let graph = graphs[at].1.clone();
                    blocks_by_class[index].passes.push((class.0.0, graph));
                }
                Stage::Run(_) | Stage::Once(OnceStage::Fill) => {
                    let graphs = &blocks_by_class[first].runs;
                    let at = shared(graphs.binary_search_by(|(sealed, _)| sealed.cmp(&class)).ok())?;
                    let graph = graphs[at].1.clone();
                    blocks_by_class[index].runs.push((class, graph));
                }
            }
        }
        let storage = block_graph_storage(&class_counts, sealed_counts)?;
        let held_storage = block_graph_storage(
            &blocks_by_class
                .iter()
                .map(|graphs| GraphCounts { runs: graphs.runs.capacity(), passes: graphs.passes.capacity() })
                .collect::<Vec<_>>(),
            sealed_counts,
        )?;
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
            least_bytes <= bytes
                && bytes <= most_bytes
                && least <= allocations
                && allocations <= most
        };
        fn heap(heap: HostHeap) -> (u64, u64) {
            (heap.bytes, heap.allocations)
        }
        if !(held_storage == storage
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
        // The runs a launch of each block class makes (as the plan certified
        // them, `certify_target_family`): both entries' and the per-layer
        // entry's of its rows, and at every block its class's runs and, where
        // the class splits there, the graphs the block runs once a group.
        let launch_runs = LaunchRuns::of(
            &served,
            blocks_by_class.iter().enumerate().flat_map(|(index, graphs)| {
                let runs = graphs.runs.iter().map(move |(staged, block)| (index, *staged, &block.plan));
                let passes = graphs
                    .passes
                    .iter()
                    .map(move |(pass_rows, block)| (index, once_class(*pass_rows, OnceStage::Pass), &block.plan));
                runs.chain(passes)
            }),
        );
        let launches = launch_classes
            .iter()
            .map(|&class| {
                let rows = class.0;
                [EntryTokens::Uploaded, EntryTokens::Selected]
                    .iter()
                    .filter_map(|&source| entries.get(&(rows, source)))
                    .map(|entry| &entry.plan)
                    .chain(per_layer_entries.get(&rows).map(|entry| &entry.plan))
                    .chain(launch_runs.of_class(class).copied())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // What a chain of each length beyond one launch further runs of
        // the graphs every split block runs once a group
        // (`chained_once_runs`), as the plan certified them.
        let chains = (2..=limits.chain_capacity.get())
            .map(|launches| {
                let further = chained_once_runs(passes, launches);
                let count = |key: (OnceStage, u64)| further.iter().find(|&&(at, _)| at == key).map_or(0, |&(_, count)| count);
                let mut runs = Vec::new();
                for graphs in &blocks_by_class {
                    for (rows, block) in &graphs.passes {
                        runs.push((block.plan.clone(), count((OnceStage::Pass, *rows))));
                    }
                    for ((class, stage), block) in &graphs.runs {
                        if *stage == Stage::Once(OnceStage::Fill) {
                            runs.push((block.plan.clone(), count((OnceStage::Fill, class.0))));
                        }
                    }
                }
                runs.retain(|&(_, count)| count > 0);
                runs
            })
            .collect::<Vec<_>>();
        // The sealed family is what the load charges; the plan's bound
        // (`TargetFamilyCertificate::graphs`), which the assessment charged
        // before any plan was sealed, holds it.
        // The launch classes ascend by rows, so those a decode step may take
        // lead them.
        let decodes = launch_classes.iter().take_while(|&&class| decode_launch(class, limits)).count();
        assert!(
            launch_classes.iter().skip(decodes).all(|&class| !decode_launch(class, limits)),
            "a family's decode launch classes lead its launch classes"
        );
        let family = NativeGraphFamily::with_launches(&plans, &launches, decodes, &chains).map_err(|error| error.to_string())?;
        let (sealed, bound) = (family.graphs(), certificate.graphs);
        let bytes = |census| device.graph_cost().bytes(census).map_or(0, |bytes| bytes.device + bytes.upload);
        tracing::info!(
            "target graphs: a decode launch holds {} bytes, the union of every launch's {} bytes; the plan bounded them at {} and {}",
            bytes(sealed.flight),
            bytes(sealed.union),
            bytes(bound.flight),
            bytes(bound.union),
        );
        assert!(
            sealed.within(&bound),
            "invariant violated: the target family's sealed graphs, {sealed:?}, exceed what the planner charged, {bound:?}"
        );
        let seal = SealReport {
            classes: launch_classes.len(),
            sealed_graphs,
            seconds: seal_seconds,
            planned,
            heap: held.heap,
            storage: held_storage,
            owned: held.owned,
            constants: held_constants,
        };
        Ok(Self {
            entries,
            blocks_by_class,
            listings,
            controls,
            upload_regions: certificate.upload_regions,
            max_segments: outline.max_segments,
            mixers: outline.mixers,
            selections: plan.blocks().iter().copied().map(slot_selection).collect(),
            family,
            max_output_bytes,
            taps: outline.tapped,
            stateless_tail,
            per_layer_entries,
            notice_rows: notice_rows(load, largest_rows),
            group: certificate_group,
            served,
            passes,
            seal,
        })
    }

    /// The stages of block `index`'s graph of a launch of `rows` rows
    /// (`block_stages`).
    pub(crate) fn stages(&self, index: usize, rows: u64) -> &'static [RunStage] {
        block_stages(&self.served, index, rows)
    }

    /// The stage of block `index`'s run of `rows` rows at `stage` of its
    /// stages: a whole decode block's resident form (`RunStage::Resident`)
    /// in a chain its bank began `resident`, where the block has one
    /// (`block_sealed_stages`), else `stage`.
    pub(crate) fn run_stage(&self, index: usize, rows: u64, stage: RunStage, resident: bool) -> RunStage {
        match stage == RunStage::Whole && resident && block_sealed_stages(&self.served, index, rows).contains(&RunStage::Resident) {
            true => RunStage::Resident,
            false => stage,
        }
    }

    /// Whether the load streams block `index`'s routed experts
    /// (`block_served`).
    pub(crate) fn streams(&self, index: usize) -> bool {
        self.served[index].streams
    }

    /// The tile counts of block `index`'s listing classes for a launch of
    /// `rows` rows, ascending; empty when that row class has none.
    pub(crate) fn listed_tiles(&self, index: usize, rows: u64) -> &[u64] {
        match self.listings.get(index) {
            Some(listings) if !crate::operators::attention::graph::decodes(rows) => &listings.tiles,
            _ => &[],
        }
    }

    /// The segment count of block `index`'s listing class for a launch over
    /// `segments` history segments: the launch's own, or the most where the
    /// block's listing classes are sealed for that alone.
    pub(crate) fn listed_segments(&self, index: usize, segments: u64) -> u64 {
        self.listings[index].segments(segments, self.max_segments)
    }

    /// Block `index`'s selected read's shape, when it reads one.
    pub(crate) fn selection(&self, index: usize) -> Option<crate::SelectionShape> {
        self.selections.get(index).copied().flatten()
    }

    /// The first row of chain region `region` of the launch-row buffers
    /// (`chain_rows`): the region the `region`th launch of a chain writes
    /// and reads.
    pub(crate) fn region_start(&self, region: usize) -> u64 {
        self.group.region_start(region)
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

    /// The most rows a group's pass covers.
    pub fn pass_rows(&self) -> u64 {
        self.passes.rows()
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub fn output_bytes(&self) -> u64 {
        self.max_output_bytes
    }

    /// The upload regions of a workspace slot
    /// (`TargetFamilyCertificate::upload_regions`): every graph seals into
    /// the layout whose input they were sized by.
    pub fn upload_regions(&self) -> &[u64] {
        &self.upload_regions
    }

    /// The controls of a launch's one upload of them.
    pub(crate) fn controls(&self) -> &LaunchControlLayout {
        &self.controls
    }
}

impl PreparedTargetGraphs {
    pub fn class_count(&self) -> usize {
        self.seal.classes
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
        // Only a load whose blocks stream their experts holds graphs of bank
        // parts and parks, which each run of a step of its bank binds
        // (`BankRun`); a bank is held exactly there.
        if resident.bank.is_some() != self.served.iter().any(|block| block.streams) {
            return Err("the resident target's expert bank disagrees with its streamed blocks".into());
        }
        let embedding = resident.embedding.table().clone();
        let direct = seismic::Tensor::from_host(
            &resident.embedding.tensor().device(),
            Element::i32(),
            &[0, 1],
            &[],
        )
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
            // The block's classes ascend in its vectors, so do its bindings.
            bound.push(BlockBindings {
                runs: bind_block_graphs(&graphs.runs, index, resident, &mut constants)?,
                passes: bind_block_graphs(&graphs.passes, index, resident, &mut constants)?,
            });
        }
        // The pass's merged experts and row table, which every group's passes
        // write and its backs read; the row table holds every first row and
        // row count, entry `i` holding `i`.
        let device = resident.embedding.tensor().device();
        let pass = match super::SplitLayers::of(&self.served) {
            None => None,
            Some(layers) => {
                let split = layers.at_rows(self.region_start(1));
                let (element, extents) = &split.merged.table;
                let counts = (0..i32::try_from(extents[0]).map_err(|_| "a pass's rows exceed i32")?)
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>();
                Some(PassBuffers {
                    experts: zeros(&device, &split.merged.experts).map_err(|error| error.to_string())?,
                    table: seismic::Tensor::from_host(&device, *element, extents, &counts)
                        .map_err(|error| format!("merged row table allocation failed: {error}"))?,
                    layers,
                })
            }
        };
        let floor = Rc::new(GroupBuffers::new(self, &device, 1, pass.as_ref()).map_err(|error| error.to_string())?);
        let per_layer = match (&resident.per_layer, self.group.per_layer) {
            (None, None) => None,
            (Some(weights), Some(extents)) => Some(BoundPerLayerEntry {
                bound: self
                    .per_layer_entries
                    .iter()
                    .map(|(rows, graph)| {
                        let absent = seismic::Tensor::from_host(
                            &weights.projection.tensor().device(),
                            Element::f32(),
                            &[0],
                            &[],
                        )
                        .map_err(|error| error.to_string())?;
                        let mut fixed = vec![
                            (&graph.ports.projection, weights.projection.tensor()),
                            (&graph.ports.norm, weights.norm.tensor()),
                            (&graph.ports.absent, &absent),
                        ];
                        if let Some(port) = &graph.ports.projection_scale {
                            let scale = weights
                                .projection
                                .scale()
                                .ok_or("the per-layer projection has no second-level scale")?;
                            fixed.push((port, scale));
                        }
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
                // The rows a single-row step leaves unwritten in the floor's
                // one region hold tokens' rows.
                token_rows: TokenRows::new(weights.table.rows(), extents[0] as usize - 1),
                table: StreamedTable::new(weights.table.clone()),
            }),
            _ => return Err("resident per-layer weights disagree with the prepared graphs".into()),
        };
        let stored = resident.sublayers.stored_blocks()?;
        let notice = self
            .notice_rows
            .map(|rows| SelectionNotice::new(&resident.embedding.tensor().device(), rows))
            .transpose()?;
        let graphs = BoundTargetGraphs {
            prepared: self.clone(),
            entry_bound,
            bound,
            constants: constants.into_tensors(),
            per_layer,
            pass,
            floor,
            held: RefCell::new(None),
            device,
            embedding,
            stored,
            notice,
            bank: resident.bank.clone(),
        };
        // A binding that holds another heap than the plan counted is a
        // defect of the plan's accounting, which refuses the load.
        let (held, planned) = (graphs.host_heap()?, self.seal.planned.bound);
        tracing::info!(
            "bound the target graphs: {} host bytes in {} allocations ({} in {} planned)",
            held.bytes, held.allocations, planned.bytes, planned.allocations
        );
        if held != planned {
            return Err(format!(
                "the bound target graphs hold {} host bytes in {} allocations; the plan counted {} in {}",
                held.bytes, held.allocations, planned.bytes, planned.allocations
            ));
        }
        Ok(graphs)
    }
}

/// The rows of each launch-row buffer (the draft input rows, the per-layer
/// rows and the kept token rows): one region of the largest row class per
/// launch a chain runs. A chain runs its launches interleaved on the device
/// (each block over every launch before the next block), so a launch's entry
/// writing these rows while an earlier launch's blocks still read its own,
/// or its blocks writing taps an earlier launch's readout has yet to read,
/// must not reach the same rows: the `k`th launch of a chain owns region `k`.
pub(super) fn chain_rows(largest_rows: u64, limits: ResourceLimits) -> Result<u64, String> {
    u64::try_from(limits.chain_capacity.get())
        .ok()
        .and_then(|regions| largest_rows.checked_mul(regions))
        .ok_or_else(|| "the chain regions' rows overflow".into())
}

/// The per-layer rows' extents over `rows` batch rows, of a per-layer entry.
pub(super) fn per_layer_rows(plan: &TargetProgramPlan, rows: u64) -> Option<[u64; 2]> {
    plan.per_layer()
        .map(|binding| [rows, binding.layers * binding.width])
}
