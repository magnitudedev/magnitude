#[cfg(test)]
use super::classes::BlockClass;
use super::{
    classes::{StagedClass, block_routed},
    heap::owned_vectors_heap,
    weights::{BankPort, GraphTimeline, WeightPort, activation},
};
use crate::{
    ModelLoadPlan, StateResourcePlan, TargetBlockProgramSlot,
    native::{AttestedTargetBlock, OutputScales},
    operators::{
        block,
        block::{
            BlockControlPorts, BlockStatePorts, FeedForwardGraph, MixerEntries, NGramParts,
            PerLayerParts,
        },
        ngram::graph::{CheckedNGramEntries, NGramStepPorts},
        paired_block,
        routed::{
            split::{OnceStage, PassPorts, RunStage, SplitPorts, Stage, StagePorts},
            streamed::{BankGraph, BankGraphEntries, CheckedBankEntries, LayerBank},
        },
        ExitMix,
        streams::{
            ResidualCarry, StreamsGeometry,
            graph::{CheckedStreamsEntries, MixingEntries, StreamsEntries, collapse, mix_site, widen},
        },
    },
    programs::{
        graph::{
            GraphError,
            draft::GraphDraft,
            tap::{TapEntry, TapPorts, TapPositions, Taps},
        },
        native_constants::{FamilyConstants, GraphConstant},
    },
};
use magnitude_family_contracts::{Decoder, ExitNorm, WeightScope};
#[cfg(test)]
use seismic::NativeGraphStorageBytes;
use seismic::{
    BackendName, Device, Element, HostHeap, NativeGraphLayout, NativeGraphMetadata,
    NativeGraphPlan, NativePort, NativeTimelinePort, WorkflowTensor,
};
use std::sync::Arc;

/// Whether the prefill graphs of `slot`'s block come in classes that list
/// their launch's history row tiles: an attention block's that may
/// (`attention::graph::lists_history_tiles`), where the plan's device of
/// `backend` reads its heads' history decoded.
fn slot_lists(
    state: &StateResourcePlan,
    backend: BackendName,
    slot: TargetBlockProgramSlot,
) -> bool {
    match slot.mixer() {
        crate::MixerProgramSlot::Attention(binding) => {
            crate::operators::attention::graph::lists_history_tiles(binding)
                && state.lists_history_tiles(backend, binding.shape.width)
        }
        _ => false,
    }
}

/// A prepared block graph and the ports of its stage (`StagePorts`): a
/// run's stage's (`P`, a split layer's front's or back's, none of a whole
/// block's or a fill's), or its group's pass's (`PassPorts`).
#[derive(Clone)]
pub(crate) struct PreparedTargetBlockGraph<P = Option<SplitPorts>> {
    /// The block it was prepared for, whose roles and layers `weights` and
    /// `banks` name.
    pub block: usize,
    pub residual: Option<WorkflowTensor>,
    pub plan: NativeGraphPlan,
    /// The carry it reads: none of a pass, which reads only crossed rows.
    pub hidden: Option<ResidualCarry<NativePort>>,
    pub weights: Vec<(WeightPort, NativePort)>,
    /// The bank ports of a block whose routed layer streams its experts.
    pub banks: Vec<(BankPort, NativePort)>,
    /// The park nodes before its calls that wait for the host, and the
    /// publications after its site requests, each with the scope of its
    /// streamed layer in the block the graph was prepared for, which each
    /// run binds (`crate::experts::Parks`).
    pub parks: Vec<(GraphTimeline, NativeTimelinePort<seismic::timeline::Parks>)>,
    /// Host constants bound statically with the weights.
    pub constants: Vec<GraphConstant>,
    /// The span marks at its ends where it is a whole block whose routed
    /// layer streams (`BlockGraphParts::span`).
    pub span: Option<(usize, usize)>,
    /// The mixer's state and control ports, of a graph that mixes (a
    /// block's back does not).
    pub mixer: Option<(BlockStatePorts, BlockControlPorts)>,
    /// The draft taps of a tapped block that this graph writes.
    pub taps: Option<TapPorts>,
    /// The per-layer rows port of a block with a per-layer input sublayer.
    pub per_layer: Option<NativePort>,
    /// What each step binds of a block with an n-gram input.
    pub ngram: Option<NGramStepPorts>,
    /// The step sequence a run writes of a graph that sets its step word
    /// (`StreamedLayer::step`).
    pub sequence: Option<NativePort>,
    /// The ports of a split block's stage of the values it crosses, which
    /// each run binds to its chain region's rows of the crossing buffers
    /// and a pass to its rows of them, and of the merged values a back
    /// combines from and a pass writes (`StagePorts`).
    pub ports: P,
    /// The graph's output: the block's new carry (the last block's exit
    /// row, which carries no pre), or a front's carry after its mixer,
    /// which its back reads as its `hidden`; a pass's merged rows.
    pub output: ResidualCarry<WorkflowTensor>,
}

/// A sealed block graph, by the ports its stage formed: a run's stage's or
/// its group's pass's.
pub(super) enum SealedBlockGraph {
    Run(Arc<PreparedTargetBlockGraph>),
    Pass(Arc<PreparedTargetBlockGraph<PassPorts>>),
}

/// A block's prepared graphs: a run's stages' graph of each class, and its
/// group's pass's of each length (`Passes::lengths`), each ascending; and a
/// tapped block's untapped run stages' graph of each class a launch none of
/// whose rows demands features takes (`FamilyOutline::untapped`), ascending,
/// which such a launch runs in place of its own. A vector of exactly each:
/// what the plan counted (`block_graph_storage`). Graphs that share a sealed
/// plan share the prepared graph.
#[derive(Clone)]
pub(super) struct BlockGraphs {
    pub(super) runs: Vec<(StagedClass, Arc<PreparedTargetBlockGraph>)>,
    pub(super) passes: Vec<(u64, Arc<PreparedTargetBlockGraph<PassPorts>>)>,
    pub(super) untapped: Vec<(StagedClass, Arc<PreparedTargetBlockGraph>)>,
}

impl SealedBlockGraph {
    pub(super) fn prepare<'a>(
        device: &Device,
        handle: &'a AttestedTargetBlock,
        tap: Option<TapEntry<'a, seismic::NativeGraph>>,
        slot: TargetBlockProgramSlot,
        inputs: BlockGraphInputs<'_>,
        layout: &NativeGraphLayout,
        family: &FamilyConstants,
    ) -> Result<Self, String> {
        let BlockGraphInputs { geometry, block_index, rows, stage, .. } = inputs;
        let mut graph = device.native_graph_with_layout(layout);
        let mut weights = Vec::new();
        let mut constants = Vec::new();
        let hidden = stage_input(geometry, block_index, stage)?
            .map(|carry| carry.try_map(|width| graph.port(Element::f32(), &[rows, width])))
            .transpose()
            .map_err(|error| error.to_string())?;
        let mixer = (&handle.mixer).into();
        let feed_forward = handle.feed_forward.as_ref().map(Into::into);
        let mut banks = Vec::new();
        let mut parks = Vec::new();
        let bank = layer_bank(stage, handle.bank.as_ref().map(BankGraphEntries::from), &mut banks, &mut parks);
        let streams = handle.streams.as_ref().map(Into::into);
        let per_layer = match (&handle.per_layer, slot.per_layer()) {
            (None, None) => None,
            (Some(kernels), Some(binding)) => Some((kernels.into(), binding)),
            _ => return Err("the block's per-layer kernels disagree with its binding".into()),
        };
        let ngram = match (&handle.ngram, slot.ngram()) {
            (None, None) => None,
            (Some(kernels), Some(binding)) => Some((kernels.into(), binding)),
            _ => return Err("the block's n-gram kernels disagree with its binding".into()),
        };
        let parts = block_graph(
            &mut graph,
            tap,
            mixer,
            FeedForwardGraph {
                entries: feed_forward,
                bank,
            },
            per_layer,
            ngram,
            streams,
            hidden.as_ref().map(|carry| carry.as_ref().map(NativePort::tensor)),
            inputs,
            &mut weights,
            &mut constants,
        )
        .map_err(|error| error.to_string())?;
        // A graph sealed once stores what it computes; a run's exports its
        // carry.
        if let Stage::Run(_) = stage {
            graph.export(&parts.output.rows).map_err(|error| error.to_string())?;
            if let Some(pre) = &parts.output.pre {
                graph.export(pre).map_err(|error| error.to_string())?;
            }
        }
        if let Some(residual) = &parts.residual {
            graph.export(residual).map_err(|error| error.to_string())?;
        }
        let plan = graph.seal().map_err(|error| error.to_string())?;
        // The graph's constants are the family's values, and its vectors
        // hold exactly their elements: what it owns then follows from its
        // topology (`owned_vectors_heap`).
        for constant in &mut constants {
            family.share(constant);
        }
        let mut mixer = parts.mixer;
        weights.shrink_to_fit();
        banks.shrink_to_fit();
        parks.shrink_to_fit();
        constants.shrink_to_fit();
        if let Some((state, _)) = &mut mixer {
            state.shrink_to_fit();
        }
        // The graph, of the ports its stage formed.
        macro_rules! graph {
            ($ports:expr) => {
                Arc::new(PreparedTargetBlockGraph {
                    block: block_index,
                    plan,
                    hidden,
                    weights,
                    banks,
                    parks,
                    constants,
                    span: parts.span,
                    mixer,
                    taps: parts.taps,
                    per_layer: parts.per_layer,
                    ngram: parts.ngram,
                    sequence: parts.sequence,
                    ports: $ports,
                    residual: parts.residual,
                    output: parts.output,
                })
            };
        }
        Ok(match parts.ports {
            StagePorts::Run(ports) => Self::Run(graph!(ports)),
            StagePorts::Pass(ports) => Self::Pass(graph!(ports)),
        })
    }
}

impl<P> PreparedTargetBlockGraph<P> {
    /// The host heap of the vectors this graph owns, at their capacities.
    /// Its plan is counted apart (`NativeGraphPlan::host_heap`), and its
    /// constants' values are the family's (`FamilyConstants`).
    pub(super) fn owned_heap(&self) -> HostHeap {
        owned_vectors_heap(
            self.weights.capacity(),
            self.banks.capacity(),
            self.parks.capacity(),
            self.constants.capacity(),
            self.mixer.as_ref().map_or(0, |(state, _)| state.ports().capacity()),
        )
    }
}

/// Everything a block graph is built from apart from its kernel entries.
#[derive(Clone, Copy)]
pub(super) struct BlockGraphInputs<'a> {
    pub(super) load: &'a ModelLoadPlan,
    pub(super) geometry: &'a Decoder,
    pub(super) state: &'a StateResourcePlan,
    pub(super) block_index: usize,
    pub(super) rows: u64,
    pub(super) segments: u64,
    pub(super) slots: u64,
    pub(super) listed: u64,
    pub(super) scored: bool,
    pub(super) output_scales: OutputScales,
    pub(super) stage: Stage,
    /// The rows a pass groups (`group_pass_rows`).
    pub(super) capacity: u64,
    /// Whether the graph exports the block's output residual beside its
    /// output: the exit block's, for a head conditioned on it
    /// (`FamilyOutline::exports_residual`).
    pub(super) residual: bool,
}

impl<'a> BlockGraphInputs<'a> {
    /// The inputs of block `block_index`'s graph of `staged`, at unit output
    /// scales: scalar arguments leave its checked shapes and storage alone.
    pub(super) fn of_class(
        load: &'a ModelLoadPlan,
        geometry: &'a Decoder,
        state: &'a StateResourcePlan,
        block_index: usize,
        ((rows, segments, slots, listed, scored), stage): StagedClass,
        capacity: u64,
    ) -> Self {
        Self {
            load,
            geometry,
            state,
            block_index,
            rows,
            segments,
            slots,
            listed,
            scored: scored > 0,
            output_scales: OutputScales { mixer: 1.0, feed_forward: 1.0, per_layer: 1.0 },
            stage,
            capacity,
            residual: false,
        }
    }
}

/// How the routed layer of a block graph of `stage` reads its experts, of a
/// block whose streamed layer's bank has `entries`: through the bank, its
/// ports and parks into `ports` and `parks`; at its home records in the
/// resident form (`RunStage::Resident`); from its stacks where the load
/// holds them whole. Only a streamed layer has a resident form
/// (`RunStage::sealed`), so a block of a held layer never has one.
fn layer_bank<'p, 'a, G: GraphDraft + 'a>(
    stage: Stage,
    entries: Option<BankGraphEntries<'a, G>>,
    ports: &'p mut Vec<(BankPort, NativePort)>,
    parks: &'p mut Vec<(GraphTimeline, NativeTimelinePort<seismic::timeline::Parks>)>,
) -> LayerBank<'p, 'a, G> {
    match (stage, entries) {
        (Stage::Run(RunStage::Resident), Some(_)) => LayerBank::Home,
        (Stage::Run(RunStage::Resident), None) => panic!("a resident form of a block whose experts are held whole"),
        (_, Some(entries)) => LayerBank::Streamed(BankGraph { entries, ports, parks }),
        (_, None) => LayerBank::Held,
    }
}

/// The widths of the carry a graph of block `block_index` of `stage` reads
/// (`ResidualCarry`): the embedded rows into block 0, and the streams
/// (`S * H`) into every later block of a decoder with residual streams and
/// into every back, which reads the carry its front's mixer left; with the
/// `[rows, S]` pre of Sinkhorn streams (none into block 0, whose first site
/// reads the one-hot pre). The pass reads only crossed rows.
pub(super) fn stage_input(geometry: &Decoder, block_index: usize, stage: Stage) -> Result<Option<ResidualCarry<u64>>, String> {
    let streams = StreamsGeometry::of(geometry).map_err(|error| error.to_string())?;
    let carry = |reads_streams: bool| match streams {
        Some(streams) if reads_streams => ResidualCarry {
            rows: streams.width(),
            pre: streams.sinkhorn().map(|_| streams.streams),
        },
        _ => ResidualCarry::rows(geometry.hidden),
    };
    Ok(match stage {
        Stage::Run(RunStage::Whole | RunStage::Resident | RunStage::Front) => Some(carry(block_index > 0)),
        Stage::Run(RunStage::Combine) => Some(carry(true)),
        Stage::Once(_) => None,
    })
}

struct BlockGraphParts {
    residual: Option<WorkflowTensor>,
    output: ResidualCarry<WorkflowTensor>,
    mixer: Option<(BlockStatePorts, BlockControlPorts)>,
    taps: Option<TapPorts>,
    /// The per-layer rows a block with a per-layer input sublayer reads.
    per_layer: Option<NativePort>,
    /// What each step binds of a block with an n-gram input.
    ngram: Option<NGramStepPorts>,
    /// The step sequence a run of a graph that sets its step word writes.
    sequence: Option<NativePort>,
    ports: StagePorts,
    /// The span marks at the ends of a whole block whose routed layer
    /// streams, which a sampled decode step's run times its block between
    /// (`BankRun::block_sampled`).
    span: Option<(usize, usize)>,
}

impl BlockGraphParts {
    /// The state ports the graph holds.
    fn state_ports(&self) -> usize {
        self.mixer.as_ref().map_or(0, |(state, _)| state.ports().len())
    }
}

/// One decoder block's graph fragment:its draft tap when tapped, then its
/// mixer then its feed-forward over `hidden`. The production graph and the
/// checked metadata route both build blocks through this one function.
#[allow(clippy::too_many_arguments)]
fn block_graph<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    tap_entry: Option<TapEntry<'a, G>>,
    mixer_entries: MixerEntries<'a, G>,
    feed_forward: FeedForwardGraph<'_, 'a, G>,
    per_layer_entries: Option<PerLayerParts<'a, G>>,
    ngram_entries: Option<NGramParts<'a, G>>,
    streams_entries: Option<StreamsEntries<'a, G>>,
    hidden: Option<ResidualCarry<&WorkflowTensor>>,
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
        capacity,
        segments,
        slots,
        listed,
        scored,
        output_scales,
        stage,
        residual,
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
        scored,
        output_scales,
    };
    let streams = StreamsGeometry::of(geometry).map_err(|error| error.to_string())?;
    if streams.is_some() != streams_entries.is_some() {
        return Err("the block's streams entries disagree with its residual form".into());
    }
    if streams.is_some() && tap_entry.is_some() {
        return Err("a draft taps residual streams".into());
    }
    // A graph sealed once is the routed layer's alone, reading no rows: a
    // streamed step's fill, or a merge set's pass over its crossed rows.
    let stage = match stage {
        Stage::Once(once) => {
            let (output, ports) = match once {
                OnceStage::Fill => (sublayers.fill(graph, feed_forward, weights, constants)?, StagePorts::Run(None)),
                OnceStage::Pass => {
                    let routed = block_routed(geometry, block_index)?.ok_or("a pass of a block without a routed layer")?;
                    let mut ports = PassPorts::of(graph, routed, geometry, rows, capacity)?;
                    let output = sublayers.pass(graph, feed_forward, &mut ports, weights, constants)?;
                    (output, StagePorts::Pass(ports))
                }
            };
            return Ok(BlockGraphParts {
                residual: None,
                output: ResidualCarry::rows(output),
                mixer: None,
                taps: None,
                per_layer: None,
                ngram: None,
                sequence: None,
                ports,
                span: None,
            });
        }
        Stage::Run(stage) => stage,
    };
    // A whole block whose routed layer streams, in either form, is marked
    // at its ends: a mark at a graph's edge adds no node and divides
    // nothing, and is stamped only on a sampled run (`seismic::NativeSpans`).
    let begun = (stage.place() == RunStage::Whole && feed_forward.bank.banked()).then(|| graph.span_mark());
    // A front writes the taps before the feed-forward, a back the output's.
    let [entry, middle, output] = stage.taps();
    let mut taps = tap_entry
        .map(|tapped| TapEntry {
            positions: TapPositions {
                entry: tapped.positions.entry && entry,
                middle: tapped.positions.middle && middle,
                output: tapped.positions.output && output,
            },
            ..tapped
        })
        .filter(|entry| entry.positions.any())
        .map(|entry| Taps::new(graph, entry, rows, geometry.hidden, activation(geometry)))
        .transpose()?;
    // A split block's crossing of its routed layer and the merged rows of
    // its merge set's pass, as this graph's ports.
    let mut split = SplitPorts::of(graph, stage, block_routed(geometry, block_index)?, geometry, rows, capacity)?;
    let hidden = hidden.ok_or("a block graph of no input rows")?;
    let hidden = ResidualCarry { rows: hidden.rows.clone(), pre: hidden.pre.cloned() };
    // A back reads the carry its front's mixer left.
    let (mixed, mut mixer) = match stage {
        RunStage::Combine => (hidden, None),
        RunStage::Whole | RunStage::Resident | RunStage::Front => {
            if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.entry) {
                taps.ports.entry = Some(taps.tap(graph, &hidden.rows)?);
            }
            // Block 0 widens the embedded rows into the streams.
            let hidden = match (streams, streams_entries) {
                (Some(streams), Some(entries)) if block_index == 0 => {
                    widen(graph, entries.repeat, streams, &hidden.rows, rows, constants)?
                }
                _ => hidden,
            };
            let (mixed, state_ports, controls) =
                sublayers.mixer(graph, mixer_entries, streams_entries, &hidden, weights, constants)?;
            if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.middle) {
                if paired.feed_forward.is_none() {
                    return Err("a draft taps the feed-forward of a lone mixer block".into());
                }
                taps.ports.middle = Some(taps.tap(graph, &mixed.rows)?);
            }
            (mixed, Some((state_ports, controls)))
        }
    };
    let (output, sequence) = sublayers.feed_forward(
        graph,
        feed_forward,
        streams_entries,
        mixed,
        weights,
        constants,
        split.as_mut().map(SplitPorts::split),
    )?;
    let ports = StagePorts::Run(split);
    // A front ends with its request: its output is the residual it mixed.
    if stage == RunStage::Front {
        return Ok(BlockGraphParts {
            residual: None,
            output,
            mixer,
            taps: taps.map(|taps| taps.ports),
            per_layer: None,
            ngram: None,
            sequence,
            ports,
            span: None,
        });
    }
    let ResidualCarry { rows: output, pre } = output;
    let (output, per_layer) =
        sublayers.per_layer(graph, per_layer_entries, output, weights, constants)?;
    // The n-gram input's windows follow the mixer's bank components; a
    // back's are its own state, over the slot tables it takes as inputs.
    let controls = mixer.as_ref().map(|(_, controls)| controls);
    let (output, ngram) =
        sublayers.ngram(graph, ngram_entries, controls, output, weights, constants)?;
    let ngram = match ngram {
        None => None,
        Some(ports) => {
            let windows = [ports.window, ports.token_window];
            match (&mut mixer, ports.controls) {
                (Some((BlockStatePorts::Recurrent(bank), _)), None) => bank.extend(windows),
                (None, Some(controls)) => {
                    mixer = Some((
                        BlockStatePorts::Recurrent(windows.into()),
                        BlockControlPorts::Recurrent(controls),
                    ));
                }
                (Some((BlockStatePorts::Attention(..), _)), _) => {
                    return Err("an n-gram input in an attention block".into());
                }
                _ => return Err("an n-gram input's slot tables disagree with its block".into()),
            }
            Some(ports.step)
        }
    };
    let residual = residual.then(|| output.clone());
    // The decoder's last block ends with its exit: a low-rank exit site,
    // whose partials the readout sums, or the collapse of Sinkhorn streams
    // by the last site's pre, which the readout normalizes.
    let last = block_index + 1 == geometry.blocks.len();
    let carry = ResidualCarry { rows: output, pre };
    let output = match (streams, streams_entries, &geometry.exit.norm) {
        (None, _, _) => carry,
        (Some(streams), Some(entries), ExitNorm::HyperConnectionHead(head)) if last => {
            let exit = ExitMix::of(head).map_err(|error| error.to_string())?;
            match (exit, entries.mixing, streams.low_rank()) {
                (ExitMix::Own(site), MixingEntries::LowRank(sites), Some(low_rank)) => {
                    let exit = sites.exit.ok_or("low-rank streams without their exit site's entries")?;
                    let partials = mix_site(
                        graph,
                        exit,
                        load,
                        WeightScope::Target,
                        low_rank,
                        &carry.rows,
                        rows,
                        site.norm.epsilon as f32,
                        false,
                        weights,
                    )?
                    .partials;
                    ResidualCarry::rows(partials)
                }
                (ExitMix::Previous(_), MixingEntries::Sinkhorn(entries), None) => {
                    ResidualCarry::rows(collapse(graph, entries.collapse, streams, &carry, rows)?)
                }
                _ => return Err("the decoder's exit disagrees with its streams' mixing".into()),
            }
        }
        (Some(_), Some(_), ExitNorm::HyperConnectionHead(_)) => carry,
        _ => return Err("residual streams without their exit".into()),
    };
    if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.output) {
        taps.ports.output = Some(taps.tap(graph, &output.rows)?);
    }
    let span = begun.map(|begun| (begun, graph.span_mark()));
    Ok(BlockGraphParts {
        residual,
        output,
        mixer,
        taps: taps.map(|taps| taps.ports),
        per_layer,
        ngram,
        sequence,
        ports,
        span,
    })
}

/// Check the production block topology from its planned entry bindings
/// without forming kernels or allocating device storage: the graph, its
/// constants, and the host heap of the vectors its prepared graph owns
/// (`owned_vectors_heap`).
pub(super) fn checked_block_graph_draft(
    mut graph: NativeGraphMetadata,
    slot: TargetBlockProgramSlot,
    tap: Option<(u64, TapPositions)>,
    inputs: BlockGraphInputs<'_>,
) -> Result<(NativeGraphMetadata, Vec<GraphConstant>, HostHeap), GraphError> {
    let BlockGraphInputs { geometry, state, block_index, rows, stage, .. } = inputs;
    let mut weights = Vec::new();
    let mut constants = Vec::new();
    let tap_elements = [("A", activation(geometry))];
    let tap_entry = tap.map(|(width, positions)| TapEntry {
        entry: &tap_elements[..],
        width,
        positions,
    });
    let hidden = stage_input(geometry, block_index, stage)?
        .map(|carry| {
            carry.try_map(|width| GraphDraft::port_with_class_extent(&mut graph, Element::f32(), &[rows, width], 0, "M"))
        })
        .transpose()?;
    let checked_streams = slot.block_streams().map(CheckedStreamsEntries::new);
    let streams = checked_streams.as_ref().map(CheckedStreamsEntries::entries);
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
    let checked_bank = slot
        .bank()
        .map(|(feed_forward, bank)| {
            CheckedBankEntries::new(feed_forward, bank, graph.backend())
        })
        .transpose()?;
    let mut banks = Vec::new();
    let mut parks = Vec::new();
    let bank = layer_bank(stage, checked_bank.as_ref().map(CheckedBankEntries::entries), &mut banks, &mut parks);
    let checked_per_layer = block::checked_per_layer(slot.per_layer())?;
    let per_layer = checked_per_layer
        .as_ref()
        .map(|(checked, binding)| (checked.entries(), *binding));
    let checked_ngram = slot
        .ngram()
        .map(|binding| (CheckedNGramEntries::new(binding), binding));
    let ngram = checked_ngram
        .as_ref()
        .map(|(checked, binding)| (checked.entries(), *binding));
    let parts = block_graph(
        &mut graph,
        tap_entry,
        mixer,
        FeedForwardGraph {
            entries: feed_forward,
            bank,
        },
        per_layer,
        ngram,
        streams,
        hidden.as_ref().map(|carry| carry.as_ref().map(NativePort::tensor)),
        inputs,
        &mut weights,
        &mut constants,
    )?;
    if let Stage::Run(_) = stage {
        GraphDraft::export(&mut graph, &parts.output.rows)?;
        if let Some(pre) = &parts.output.pre {
            GraphDraft::export(&mut graph, pre)?;
        }
    }
    if let Some(residual) = &parts.residual {
        GraphDraft::export(&mut graph, residual)?;
    }
    let owned = owned_vectors_heap(
        weights.len(),
        banks.len(),
        parks.len(),
        constants.len(),
        parts.state_ports(),
    );
    Ok((graph, constants, owned))
}

#[cfg(test)]
pub(crate) fn checked_block_graph_resources(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    slot: TargetBlockProgramSlot,
    block_index: usize,
    class: BlockClass,
) -> Result<(NativeGraphStorageBytes, Vec<GraphConstant>), GraphError> {
    let staged = (class, Stage::Run(RunStage::Whole));
    let (graph, constants, _) = checked_block_graph_draft(
        NativeGraphMetadata::new(backend),
        slot,
        None,
        BlockGraphInputs::of_class(load, geometry, state, block_index, staged, class.0),
    )?;
    Ok((GraphDraft::seal(graph)?, constants))
}
