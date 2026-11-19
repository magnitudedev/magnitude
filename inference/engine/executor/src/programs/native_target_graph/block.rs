#[cfg(test)]
use super::classes::BlockClass;
use super::{
    classes::{StagedClass, block_routed},
    heap::owned_vectors_heap,
    weights::{BankPort, BlockWeights, GraphTimeline, GraphWeights, PlannedWeights, WeightPart, WeightPort, activation, weight_port},
};
use crate::{
    MixerProgramSlot, ModelLoadPlan, StateResourcePlan, TargetBlockProgramSlot,
    native::{AttestedTargetBlock, OutputScales},
    operators::{
        ExitMix, block,
        block::{
            BlockState, FeedForwardGraph, MixerEntries, SublayerGraph, SublayersNumbers,
            SublayersTopology,
        },
        ngram::graph::{CheckedNGramEntries, NGramEntries, NGramStepPorts},
        paired_block,
        per_layer::graph::{CheckedPerLayerEntries, PerLayerEntries},
        routed::{
            fused_graph::OperandOut,
            split::{self, OnceStage, PassPorts, RunStage, SplitBuild, SplitForm, SplitLayer, SplitPorts, Stage},
            streamed::{BankGraph, BankGraphEntries, CheckedBankEntries, LayerBank},
        },
        streams::{
            ResidualCarry, SinkhornExit, StreamsGeometry, StreamsMixing,
            graph::{CheckedStreamsEntries, MixingEntries, StreamsEntries, collapse, mix_site, widen},
        },
    },
    programs::{
        graph::{
            GraphError,
            draft::{Certified, GraphDraft},
            facts::{FactIndex, FactSink, Facts, Numbers, RealIndex, TableIndex, constant, constant_reals},
            tap::{TapEntry, TapPorts, TapPositions, Taps},
        },
        native_constants::{FamilyConstants, GraphConstant, RecordedConstant},
    },
};
use magnitude_family_contracts::{Decoder, ExitNorm, SublayerIndex, WeightKind, WeightRole, WeightScope};
#[cfg(test)]
use seismic::NativeGraphStorageBytes;
use seismic::{
    BackendName, ClassFact, Device, Dim, Element, HostHeap, NativeGraphClassSlice, NativeGraphLayout,
    NativeGraphMetadata, NativeGraphPlan, NativeGraphResourceTemplate, NativeGraphTraces, NativePort,
    NativeTimelinePort, OperandCarry, OperandForms, Reals, Table, TraceOutput, WorkflowTensor, Argument,
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
                && state.lists_history_tiles(backend, binding.shape.widths.width)
        }
        _ => false,
    }
}

/// The rows of the one class an operand carry exists in: a decode step's,
/// whose block graphs run whole (`BlockGraphInputs::split`). The carry's
/// forms are its reader's at these rows, one choice there.
pub(super) const CARRIED_ROWS: u64 = 1;

/// One block boundary's operand carry at the decode class
/// (`seismic::OperandCarry`), as the graphs on both of its sides declare it
/// (`block_carries`): its reader's input norm's resident element, and the
/// epsilon the reader normalizes under, as its projection reads it (an
/// `f32`, held by its bits). Block L's fused routed output forms block
/// L+1's operand into the carry's ports, and block L+1's projection reads
/// it there in place of its formation launch.
///
/// The forms the operand is carried in (`seismic::OperandForms`) are the
/// reader's kernel's at the carried rows, a tuning outcome. `R` is what a
/// side holds of them, one of three by what is known where it stands:
/// - the reader's mixer binding (`MixerProgramSlot`), in a family's
///   outline: a load prepares one set of kernels per binding
///   (`AttestedTarget`), so sides naming one reader are carried in one
///   form, whichever it is, and blocks share a sealed graph only over
///   equal readers (`BlockGraphShape`);
/// - nothing (`()`), in a graph built to be sized: a carry's ports and
///   everything a graph reserves for it are the same in every form;
/// - the forms themselves, in a graph of kernels, whose seal writes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CarrySide<R = ()> {
    reader: R,
    norm: Element,
    epsilon: u32,
}

impl<R> CarrySide<R> {
    pub(super) fn new(reader: R, norm: Element, epsilon: f32) -> Self {
        Self { reader, norm, epsilon: epsilon.to_bits() }
    }

    fn epsilon(&self) -> f32 {
        f32::from_bits(self.epsilon)
    }

    /// The side holding `reader` of its reader's forms.
    fn reading<S>(self, reader: S) -> CarrySide<S> {
        CarrySide { reader, norm: self.norm, epsilon: self.epsilon }
    }
}

/// The operand carries of a block's decode graphs (`block_carries`): the
/// one it reads, formed by the block before (none into block 0), and the
/// one it writes, which the block after reads (none out of the last block).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BlockCarries<R = ()> {
    pub(super) carry_in: Option<CarrySide<R>>,
    pub(super) carry_out: Option<CarrySide<R>>,
}

impl<R> Default for BlockCarries<R> {
    fn default() -> Self {
        Self { carry_in: None, carry_out: None }
    }
}

impl<R> BlockCarries<R> {
    /// The carries as a graph built to be sized declares them: in no form.
    fn sized(self) -> BlockCarries {
        BlockCarries {
            carry_in: self.carry_in.map(|side| side.reading(())),
            carry_out: self.carry_out.map(|side| side.reading(())),
        }
    }
}

impl BlockCarries {
    /// The carries in the forms `forms` gives of each one's reader, the
    /// block that many after this one: 0 its own, which reads the carry in;
    /// 1 the next, which reads the carry out (as `carried_norm` names
    /// their norms).
    fn formed<F, E>(self, mut forms: impl FnMut(u32) -> Result<F, E>) -> Result<BlockCarries<F>, E> {
        let mut formed = |side: Option<CarrySide>, reader: u32| {
            side.map(|side| Ok::<_, E>(side.reading(forms(reader)?))).transpose()
        };
        Ok(BlockCarries { carry_in: formed(self.carry_in, 0)?, carry_out: formed(self.carry_out, 1)? })
    }
}

/// The input norm a carry's operand is formed under, as a block graph
/// names it relative to its block (`relative_scope`): the reading block's
/// own (block 0), or the next block's (block 1) in the graph that forms it.
/// Through `block_scope`, the two sides of one boundary name one weight.
pub(super) fn carried_norm(block: u32) -> WeightPort {
    WeightPort {
        role: WeightRole {
            scope: WeightScope::TargetSublayer(SublayerIndex { block, sublayer: 0 }),
            kind: WeightKind::InputNorm,
        },
        part: WeightPart::Values,
    }
}

/// An operand carry a block graph declares: its ports, which each run
/// binds, the forms `F` its graph names it in (`GraphDraft::Forms`), and
/// the norm weight port its operand is formed under.
#[derive(Clone)]
pub(crate) struct CarryPorts<F = OperandForms> {
    pub carry: OperandCarry,
    pub forms: F,
    pub norm: WeightPort,
}

/// Certify the boundary between block `index`'s decode graph `from` and
/// block `index + 1`'s decode graph `to`, which a step runs one after the
/// other in either form each block has there (tapped or untapped, whole or
/// resident): `from` carries out exactly where `to` carries in, at one
/// activation, and the norms the two name relative to their blocks are,
/// through `block_scope`, one weight: block `index + 1`'s input norm.
///
/// # Panics
///
/// Where `from` writes the operand in other forms than `to` reads it in.
/// Each sealed its side in the forms of the kernels of its carry's reader
/// (`BlockCarries::formed`), and blocks hold one sealed graph only over
/// equal readers (`BlockGraphShape`), whose kernels are one preparation's:
/// two forms at one boundary mean kernels of one binding that differ.
pub(super) fn certify_carry<P, Q>(
    index: usize,
    from: &PreparedTargetBlockGraph<P>,
    to: &PreparedTargetBlockGraph<Q>,
) -> Result<(), String> {
    let next = index + 1;
    let (out, into) = match (&from.operand, &to.operand_in) {
        (None, None) => return Ok(()),
        (Some(out), Some(into)) => (out, into),
        (out, into) => {
            return Err(format!(
                "block {index}'s decode graph carries out {} operand, block {next}'s carries in {}",
                if out.is_some() { "an" } else { "no" },
                if into.is_some() { "one" } else { "none" },
            ));
        }
    };
    let named = |ports: &CarryPorts, block: usize| {
        let role = ports.norm.role;
        super::prepared::block_scope(role.scope, block).map(|scope| WeightPort { role: WeightRole { scope, ..role }, part: ports.norm.part })
    };
    let (formed, read) = (named(out, index)?, named(into, next)?);
    let own = WeightPort {
        role: WeightRole { scope: super::prepared::block_scopes(next)?[0], kind: WeightKind::InputNorm },
        part: WeightPart::Values,
    };
    if formed != read || read != own {
        return Err(format!(
            "block {index} forms its carried operand under {formed:?}, block {next} reads it under {read:?}, not its input norm"
        ));
    }
    if out.carry.activation != into.carry.activation {
        return Err(format!(
            "block {index} carries an operand at {:?}, block {next} reads one at {:?}",
            out.carry.activation, into.carry.activation
        ));
    }
    assert_eq!(
        out.forms, into.forms,
        "block {index} carries its operand out in other forms than block {next}'s kernels read it in"
    );
    Ok(())
}

/// A prepared block graph and the ports of its stage (`StagePorts`): a
/// run's stage's (`P`, a split layer's front's or back's, none of a whole
/// block's or a fill's), or its group's pass's (`PassGraph`). Its weights,
/// banks and parks name their scopes relative to its block
/// (`relative_scope`), so equal-shape blocks share it, each binding its own
/// (`block_scope`).
#[derive(Clone)]
pub(crate) struct PreparedTargetBlockGraph<P = Option<SplitPorts>> {
    pub residual: Option<WorkflowTensor>,
    pub plan: NativeGraphPlan,
    /// The carry it reads: none of a pass, which reads only crossed rows.
    pub hidden: Option<ResidualCarry<NativePort>>,
    pub weights: Vec<(WeightPort, NativePort)>,
    /// The bank ports of a block whose routed layer streams its experts.
    pub banks: Vec<(BankPort, NativePort)>,
    /// The park nodes before its calls that wait for the host, and the
    /// publications after its site requests, each with the scope of its
    /// streamed layer relative to the block (`relative_scope`), which each
    /// run binds (`crate::experts::Parks`).
    pub parks: Vec<(GraphTimeline, NativeTimelinePort<seismic::timeline::Parks>)>,
    /// Host constants bound statically with the weights.
    pub constants: Vec<GraphConstant>,
    /// The span marks at its ends where it is a whole block whose routed
    /// layer streams (`BlockGraphParts::span`).
    pub span: Option<(usize, usize)>,
    /// The block's state and control ports: its mixer's history or bank,
    /// and its n-gram input's bank (a block's back keeps only the latter).
    pub state: BlockState,
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
    /// The next block's operand its output forms (`BlockCarries::carry_out`).
    pub operand: Option<CarryPorts>,
    /// The operand its mixer's projection reads, formed by the block before
    /// (`BlockCarries::carry_in`).
    pub operand_in: Option<CarryPorts>,
}

/// A sealed block graph, by the ports its stage formed: a run's stage's or
/// its group's pass's.
pub(super) enum SealedBlockGraph {
    Run(Arc<PreparedTargetBlockGraph>),
    Pass(Arc<PreparedTargetBlockGraph<PassGraph>>),
}

/// The ports of a block graph of a stage ([`Stage`]): a run's, which a
/// split layer's front or back crosses or merges values through (none of a
/// whole block's or a fill's), or its merge set's pass's, with the expert
/// tiles its merged experts hold. A prepared graph is held by which of the
/// two it has, so a run's graph and a pass's are apart by type.
enum StagePorts {
    Run(Option<SplitPorts>),
    Pass(PassPorts, Dim),
}

/// A prepared pass's ports, and the expert tiles of the merged experts it
/// writes, as its graph was built: what each pass binds of the group's
/// merged buffers.
#[derive(Clone)]
pub(crate) struct PassGraph {
    pub ports: PassPorts,
    pub blocks: u64,
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
    pub(super) passes: Vec<(u64, Arc<PreparedTargetBlockGraph<PassGraph>>)>,
    pub(super) untapped: Vec<(StagedClass, Arc<PreparedTargetBlockGraph>)>,
}

impl SealedBlockGraph {
    /// The block graph `inputs` names, built from its split at the model's
    /// own numbers over the load's weights into `layout`, its operand
    /// carries in the forms `forms` gives of each one's reader's kernels
    /// (`BlockCarries::formed`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare<'a>(
        device: &Device,
        handle: &'a AttestedTargetBlock,
        tap: Option<(u64, TapEntry<'a, seismic::NativeGraph>)>,
        slot: TargetBlockProgramSlot,
        inputs: BlockGraphInputs<'_>,
        forms: impl FnMut(u32) -> Result<OperandForms, String>,
        layout: &NativeGraphLayout,
        family: &FamilyConstants,
    ) -> Result<Self, String> {
        let stage = inputs.stage;
        let text = |error: GraphError| error.to_string();
        let planned = inputs.weights().map_err(text)?;
        let mut graph = device.native_graph_with_layout(layout);
        let mut weights = Vec::new();
        let mut constants = Vec::new();
        let mut banks = Vec::new();
        let mut parks = Vec::new();
        let bank = layer_bank(stage, handle.bank.as_ref().map(BankGraphEntries::from), &mut banks, &mut parks);
        let (shape, numbers) = inputs
            .constant_split(slot, &bank, tap.as_ref().map(|(width, tap)| (*width, tap.positions)), forms)
            .map_err(text)?;
        let hidden = numbers
            .input
            .clone()
            .map(|carry| carry.try_map(|width| GraphDraft::port(&mut graph, Element::f32(), &[&numbers.sublayers.rows, &width])))
            .transpose()
            .map_err(|error| error.to_string())?;
        let entries = BlockEntries {
            tap: tap.map(|(_, tap)| tap),
            mixer: (&handle.mixer).into(),
            feed_forward: FeedForwardGraph { entries: handle.feed_forward.as_ref().map(Into::into), bank },
            per_layer: handle.per_layer.as_ref().map(Into::into),
            ngram: handle.ngram.as_ref().map(Into::into),
            streams: handle.streams.as_ref().map(Into::into),
        };
        let parts = block_graph(
            &mut graph,
            entries,
            hidden.as_ref().map(ResidualCarry::as_ref),
            &shape,
            &numbers,
            &planned,
            &mut weights,
            &mut constants,
        )
        .map_err(text)?;
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
        let mut state = parts.state;
        weights.shrink_to_fit();
        banks.shrink_to_fit();
        parks.shrink_to_fit();
        constants.shrink_to_fit();
        state.shrink_to_fit();
        // The graph, of the ports its stage formed.
        macro_rules! graph {
            ($ports:expr) => {
                Arc::new(PreparedTargetBlockGraph {
                    plan,
                    hidden,
                    weights,
                    banks,
                    parks,
                    constants,
                    span: parts.span,
                    state,
                    taps: parts.taps,
                    per_layer: parts.per_layer,
                    ngram: parts.ngram,
                    sequence: parts.sequence,
                    ports: $ports,
                    residual: parts.residual,
                    output: parts.output,
                    operand: parts.operand,
                    operand_in: parts.operand_in,
                })
            };
        }
        Ok(match parts.ports {
            StagePorts::Run(ports) => Self::Run(graph!(ports)),
            StagePorts::Pass(ports, blocks) => Self::Pass(graph!(PassGraph { ports, blocks: blocks.fixed() })),
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
            self.state.capacities(),
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
    /// The block's operand carries (`FamilyOutline::carries`), which its
    /// graphs of a one-row class declare.
    pub(super) carries: BlockCarries<MixerProgramSlot>,
}

impl<'a> BlockGraphInputs<'a> {
    /// The inputs of block `block_index`'s graph of `staged`, at unit output
    /// scales: scalar arguments leave its checked shapes and storage alone.
    /// The block carries `carries` (`FamilyOutline::carries`).
    pub(super) fn of_class(
        load: &'a ModelLoadPlan,
        geometry: &'a Decoder,
        state: &'a StateResourcePlan,
        block_index: usize,
        ((rows, segments, slots, listed, scored), stage): StagedClass,
        capacity: u64,
        carries: BlockCarries<MixerProgramSlot>,
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
            carries,
        }
    }
}

/// What a block graph's builder branches on apart from its kernel entries
/// (`BlockGraphInputs::split`): its stage, whether it exports its residual,
/// its draft taps' positions and element, its routed layer's crossing, the
/// decoder exit it ends with (the last block of residual streams), its
/// sublayers' and its operand carries, in the forms `F` its graph names
/// them in (`GraphDraft::Forms`). No model number.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct BlockShape<R = Argument<f64>, T = Table, F = ()> {
    stage: Stage,
    /// Block 0, which widens the embedded rows into residual streams.
    first: bool,
    residual: bool,
    tap: Option<TapPositions>,
    activation: Element,
    split: Option<SplitForm>,
    exit: Option<ExitTopology<R>>,
    sublayers: SublayersTopology<R, T>,
    /// The operand carries it declares: the block's
    /// (`BlockGraphInputs::carries`) in a whole run of one row, the decode
    /// step's, where the consumer's projection pairs (`M == 1`); none in
    /// every other graph.
    carries: BlockCarries<F>,
}

impl<R, T, F> BlockShape<R, T, F> {
    /// It with its tables `f` of its own (`SublayersTopology::tables`).
    fn tables<U>(self, f: impl FnMut(T) -> U) -> BlockShape<R, U, F> {
        BlockShape {
            stage: self.stage,
            first: self.first,
            residual: self.residual,
            tap: self.tap,
            activation: self.activation,
            split: self.split,
            exit: self.exit,
            sublayers: self.sublayers.tables(f),
            carries: self.carries,
        }
    }
}

impl<R, T> BlockShape<R, T> {
    /// It as a graph of kernels declares it: its carries in the forms
    /// `forms` gives (`BlockCarries::formed`).
    fn formed<F, E>(self, forms: impl FnMut(u32) -> Result<F, E>) -> Result<BlockShape<R, T, F>, E> {
        Ok(BlockShape {
            stage: self.stage,
            first: self.first,
            residual: self.residual,
            tap: self.tap,
            activation: self.activation,
            split: self.split,
            exit: self.exit,
            sublayers: self.sublayers,
            carries: self.carries.formed(forms)?,
        })
    }
}

impl<R, T, F> Numbers<R> for BlockShape<R, T, F> {
    type Of<E> = BlockShape<E, T, F>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> BlockShape<E, T, F> {
        BlockShape {
            stage: self.stage,
            first: self.first,
            residual: self.residual,
            tap: self.tap,
            activation: self.activation,
            split: self.split,
            exit: self.exit.map(|exit| match exit {
                ExitTopology::LowRank { epsilon } => ExitTopology::LowRank { epsilon: f(epsilon) },
                ExitTopology::Collapse(exit) => ExitTopology::Collapse(exit),
            }),
            sublayers: self.sublayers.map(f),
            carries: self.carries,
        }
    }
}

/// The decoder's exit at the end of the last block of residual streams: a
/// low-rank exit site at its norm's epsilon, whose partials the readout
/// sums, or the collapse of Sinkhorn streams by the last site's pre or into
/// their mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ExitTopology<R> {
    LowRank { epsilon: R },
    Collapse(SinkhornExit),
}

/// Every number a block graph is sized by: the rows a pass groups, its
/// draft input rows' width, the carry it reads, its routed layer's
/// crossing and its sublayers'.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct BlockNumbers<D = Dim> {
    capacity: D,
    tap: Option<D>,
    input: Option<ResidualCarry<D>>,
    split: Option<SplitLayer<D>>,
    sublayers: SublayersNumbers<D>,
}

impl<D> Numbers<D> for BlockNumbers<D> {
    type Of<E> = BlockNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> BlockNumbers<E> {
        BlockNumbers {
            capacity: f(self.capacity),
            tap: self.tap.map(&mut *f),
            input: self.input.map(|carry| ResidualCarry { rows: f(carry.rows), pre: carry.pre.map(&mut *f) }),
            split: self.split.map(|layer| layer.map(&mut *f)),
            sublayers: self.sublayers.map(f),
        }
    }
}

impl<'a> BlockGraphInputs<'a> {
    /// The block's index as its scopes name it.
    fn block(&self) -> Result<u32, GraphError> {
        Ok(u32::try_from(self.block_index).map_err(|_| "target block index exceeds u32")?)
    }

    /// The load's weights as the block's graph reads them (`BlockWeights`).
    fn weights(&self) -> Result<BlockWeights<'a>, GraphError> {
        Ok(BlockWeights { load: self.load, block: self.block()? })
    }

    /// The block graph as its builder branches on it and the numbers it is
    /// sized by, its routed layer reading its experts as `bank` says and
    /// its draft taps (`tap`: the draft input rows' width and the
    /// positions) as given: the one place its model numbers are read.
    fn split<G: GraphDraft>(
        &self,
        slot: TargetBlockProgramSlot,
        bank: &LayerBank<'_, '_, G>,
        tap: Option<(u64, TapPositions)>,
    ) -> Result<(BlockShape<f64, Arc<[u8]>>, BlockNumbers<u64>), GraphError> {
        let Self {
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
            stage,
            capacity,
            residual,
            carries,
        } = *self;
        let block = geometry.blocks.get(block_index).ok_or("target block geometry is absent")?;
        let paired = paired_block(block).map_err(|error| error.to_string())?;
        let (sublayers, numbers) = block::BlockSublayers {
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
        }
        .split(bank, slot.per_layer(), slot.ngram())?;
        if numbers.streams.is_some() && tap.is_some() {
            return Err("a draft taps residual streams".into());
        }
        let last = block_index + 1 == geometry.blocks.len();
        let exit = match (&numbers.streams, &geometry.exit.norm) {
            (None, _) => None,
            (Some(streams), ExitNorm::HyperConnectionHead(head)) if last => {
                Some(match (ExitMix::of(head).map_err(|error| error.to_string())?, streams.mixing) {
                    (ExitMix::Own(site), StreamsMixing::LowRank { .. }) => {
                        ExitTopology::LowRank { epsilon: site.norm.epsilon }
                    }
                    (mix, StreamsMixing::Sinkhorn { .. }) => ExitTopology::Collapse(
                        SinkhornExit::of(mix).ok_or("the decoder's exit disagrees with its streams' mixing")?,
                    ),
                    _ => return Err("the decoder's exit disagrees with its streams' mixing".into()),
                })
            }
            (Some(_), ExitNorm::HyperConnectionHead(_)) => None,
            (Some(_), _) => return Err("residual streams without their exit".into()),
        };
        let split = block_routed(geometry, block_index)?.map(|routed| split::split(routed, geometry));
        // Both sides of a boundary run each decode step at one row, whole,
        // in either form of a streamed block (`RunStage::Resident`).
        let carries = match stage {
            Stage::Run(run) if rows == CARRIED_ROWS && run.place() == RunStage::Whole => carries.sized(),
            _ => BlockCarries::default(),
        };
        Ok((
            BlockShape {
                stage,
                first: block_index == 0,
                residual,
                tap: tap.map(|(_, positions)| positions),
                activation: activation(geometry),
                split: split.as_ref().map(|(form, _)| *form),
                exit,
                sublayers,
                carries,
            },
            BlockNumbers {
                capacity,
                tap: tap.map(|(width, _)| width),
                input: stage_input(geometry, block_index, stage)?,
                split: split.map(|(_, layer)| layer),
                sublayers: numbers,
            },
        ))
    }

    /// The block graph's split (`split`) as a graph is built from it at the
    /// model's own numbers: its reals and tables constant, its carries in
    /// the forms `forms` gives (`BlockCarries::formed`), its numbers
    /// constant. Each whole conversion of the split stays in this frame.
    fn constant_split<G: GraphDraft>(
        &self,
        slot: TargetBlockProgramSlot,
        bank: &LayerBank<'_, '_, G>,
        tap: Option<(u64, TapPositions)>,
        forms: impl FnMut(u32) -> Result<G::Forms, String>,
    ) -> Result<(BlockShape<Argument<f64>, Table, G::Forms>, BlockNumbers), GraphError> {
        let (shape, numbers) = self.split(slot, bank, tap)?;
        Ok((constant_reals(shape).tables(Table::from).formed(forms)?, constant(numbers)))
    }

    /// The block graph's split (`split`) as its topology key holds it: its
    /// numbers as facts of `facts`, each class dimension one fact, the
    /// block's, which its classes vary, wherever an operator reads it
    /// (`SublayersNumbers::share_classes`); its reals and tables indexed.
    /// Each whole conversion of the split stays in this frame.
    fn indexed_split<G: GraphDraft>(
        &self,
        slot: TargetBlockProgramSlot,
        bank: &LayerBank<'_, '_, G>,
        tap: Option<(u64, TapPositions)>,
        facts: &mut FactSink,
    ) -> Result<(BlockShape<RealIndex, TableIndex>, BlockNumbers<FactIndex>), GraphError> {
        let (shape, numbers) = self.split(slot, bank, tap)?;
        let mut numbers = facts.index(numbers);
        numbers.sublayers = numbers.sublayers.share_classes();
        Ok((facts.index_reals(shape).tables(|bytes| facts.table(bytes)), numbers))
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
/// `[rows, S]` pre of Sinkhorn streams mixing by the previous site's (none
/// into block 0, whose first site reads the one-hot pre). The pass reads
/// only crossed rows.
pub(super) fn stage_input(geometry: &Decoder, block_index: usize, stage: Stage) -> Result<Option<ResidualCarry<u64>>, String> {
    let streams = StreamsGeometry::of(geometry).map_err(|error| error.to_string())?;
    let carry = |reads_streams: bool| match streams {
        Some(streams) if reads_streams => ResidualCarry {
            rows: streams.width(),
            pre: streams.carried_pre(),
        },
        _ => ResidualCarry::rows(geometry.hidden),
    };
    Ok(match stage {
        Stage::Run(RunStage::Whole | RunStage::Resident | RunStage::Front) => Some(carry(block_index > 0)),
        Stage::Run(RunStage::Combine) => Some(carry(true)),
        Stage::Once(_) => None,
    })
}

struct BlockGraphParts<F> {
    residual: Option<WorkflowTensor>,
    output: ResidualCarry<WorkflowTensor>,
    state: BlockState,
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
    /// The operand carries it declares (`BlockShape::carries`).
    operand: Option<CarryPorts<F>>,
    operand_in: Option<CarryPorts<F>>,
}

impl<F> BlockGraphParts<F> {
    /// The state ports the graph holds.
    fn state_ports(&self) -> [usize; 2] {
        [
            self.state.history.as_ref().map_or(0, |history| history.planes.len()),
            self.state.bank.as_ref().map_or(0, |bank| bank.components.len()),
        ]
    }
}

/// A block graph's kernel entries: prepared kernels or checked bindings.
struct BlockEntries<'p, 'a, G: GraphDraft + 'a> {
    tap: Option<TapEntry<'a, G>>,
    mixer: MixerEntries<'a, G>,
    feed_forward: FeedForwardGraph<'p, 'a, G>,
    per_layer: Option<PerLayerEntries<'a, G>>,
    ngram: Option<NGramEntries<'a, G>>,
    streams: Option<StreamsEntries<'a, G>>,
}

/// One decoder block's graph fragment, built from its split alone (`shape`
/// at `numbers`, over `planned`): its draft tap when tapped, then its mixer
/// then its feed-forward over the carry bound to the ports `hidden`. The
/// production graph and the recorded metadata route both build blocks
/// through this one function.
#[allow(clippy::too_many_arguments)]
fn block_graph<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: BlockEntries<'_, 'a, G>,
    hidden: Option<ResidualCarry<&NativePort>>,
    shape: &BlockShape<Argument<f64>, Table, G::Forms>,
    numbers: &BlockNumbers,
    planned: &(impl PlannedWeights + ?Sized),
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<BlockGraphParts<G::Forms>, GraphError> {
    let BlockEntries { tap: tap_entry, mixer: mixer_entries, feed_forward, per_layer, ngram, streams: streams_entries } =
        entries;
    // A target block's graph names its own block as block 0, the others
    // relative to it (`relative_scope`).
    let sublayers = SublayerGraph { block: 0, topology: &shape.sublayers, numbers: &numbers.sublayers, weights: planned };
    let rows = &numbers.sublayers.rows;
    let streams = numbers.sublayers.streams.as_ref();
    if streams.is_some() != streams_entries.is_some() {
        return Err("the block's streams entries disagree with its residual form".into());
    }
    // A graph sealed once is the routed layer's alone, reading no rows: a
    // streamed step's fill, or a merge set's pass over its crossed rows.
    let stage = match shape.stage {
        Stage::Once(once) => {
            let (output, ports) = match once {
                OnceStage::Fill => (sublayers.fill(graph, feed_forward, weights, constants)?, StagePorts::Run(None)),
                OnceStage::Pass => {
                    let (Some(form), Some(layer)) = (&shape.split, &numbers.split) else {
                        return Err("a pass of a block without a routed layer".into());
                    };
                    let (mut ports, blocks) = PassPorts::of(graph, form, layer, rows, &numbers.capacity)?;
                    let output = sublayers.pass(graph, feed_forward, &mut ports, &blocks, weights, constants)?;
                    (output, StagePorts::Pass(ports, blocks))
                }
            };
            return Ok(BlockGraphParts {
                residual: None,
                output: ResidualCarry::rows(output),
                state: BlockState::default(),
                taps: None,
                per_layer: None,
                ngram: None,
                sequence: None,
                ports,
                span: None,
                operand: None,
                operand_in: None,
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
        .map(|entry| {
            let width = numbers.tap.as_ref().ok_or("a tapped block without its draft input width")?;
            Taps::new(graph, entry, width, rows, &numbers.sublayers.hidden, shape.activation)
        })
        .transpose()?;
    // A split block's crossing of its routed layer and the merged rows of
    // its merge set's pass, as this graph's ports.
    let split_layer = shape.split.as_ref().zip(numbers.split.as_ref());
    let mut split = SplitPorts::of(graph, stage, split_layer, rows, &numbers.capacity)?;
    let hidden_ports = hidden.ok_or("a block graph of no input rows")?;
    let hidden = ResidualCarry {
        rows: hidden_ports.rows.tensor().clone(),
        pre: hidden_ports.pre.map(|pre| pre.tensor().clone()),
    };
    let carries = shape.carries;
    // A carry's operand is of the block's input or output rows alone: none
    // of residual streams, nor of an output a later sublayer or the exit
    // changes after the feed-forward.
    if carries.carry_out.is_some()
        && (streams.is_some()
            || shape.exit.is_some()
            || shape.sublayers.per_layer.is_some()
            || shape.sublayers.ngram.is_some())
    {
        return Err("a block carries its output's operand past a later sublayer".into());
    }
    if carries.carry_in.is_some() && streams.is_some() {
        return Err("a block reads a carried operand of residual streams".into());
    }
    let mut operand_in = None;
    // A back reads the carry its front's mixer left.
    let (mixed, mut state_ports) = match stage {
        RunStage::Combine => (hidden, BlockState::default()),
        RunStage::Whole | RunStage::Resident | RunStage::Front => {
            if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.entry) {
                taps.ports.entry = Some(taps.tap(graph, &hidden.rows)?);
            }
            // Block 0 widens the embedded rows into the streams.
            let hidden = match (streams, streams_entries) {
                (Some(streams), Some(entries)) if shape.first => {
                    widen(graph, entries.repeat, streams.clone(), &hidden.rows, rows.clone(), constants)?
                }
                _ => hidden,
            };
            let (mixed, state_ports) =
                sublayers.mixer(graph, mixer_entries, streams_entries, &hidden, weights, constants)?;
            // The mixer's projection reads the operand the block before
            // formed under this block's input norm, the port it bound.
            if let Some(side) = carries.carry_in {
                let norm = carried_norm(0);
                let port = weights
                    .iter()
                    .find_map(|(weight, port)| (*weight == norm).then_some(port))
                    .ok_or("a carried operand's reader binds no input norm")?
                    .clone();
                let carry = graph.operand_carry(CARRIED_ROWS, &numbers.sublayers.hidden, shape.activation)?;
                graph.carry_in(hidden_ports.rows, &carry, side.reader, &port, side.epsilon())?;
                operand_in = Some(CarryPorts { carry, forms: side.reader, norm });
            }
            if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.middle) {
                if shape.sublayers.feed_forward.is_none() {
                    return Err("a draft taps the feed-forward of a lone mixer block".into());
                }
                taps.ports.middle = Some(taps.tap(graph, &mixed.rows)?);
            }
            (mixed, state_ports)
        }
    };
    // The output forms the next block's operand under that block's input
    // norm, which this graph binds (`carried_norm`).
    let operand = carries
        .carry_out
        .map(|side| {
            let norm = carried_norm(1);
            let port = weight_port(graph, planned, norm.role, norm.part, weights)?;
            let carry = graph.operand_carry(CARRIED_ROWS, &numbers.sublayers.hidden, shape.activation)?;
            Ok::<_, GraphError>((side, CarryPorts { carry, forms: side.reader, norm }, port))
        })
        .transpose()?;
    let (output, sequence) = sublayers.feed_forward(
        graph,
        feed_forward,
        streams_entries,
        mixed,
        weights,
        constants,
        split.as_mut().map(SplitBuild::split),
        operand.as_ref().map(|(side, ports, norm)| OperandOut {
            carry: &ports.carry,
            forms: side.reader,
            norm: norm.tensor(),
            epsilon: side.epsilon(),
        }),
    )?;
    let operand = operand.map(|(_, ports, _)| ports);
    let ports = StagePorts::Run(split.map(SplitBuild::into_ports));
    // A front ends with its request: its output is the residual it mixed.
    if stage == RunStage::Front {
        return Ok(BlockGraphParts {
            residual: None,
            output,
            state: state_ports,
            taps: taps.map(|taps| taps.ports),
            per_layer: None,
            ngram: None,
            sequence,
            ports,
            span: None,
            operand,
            operand_in,
        });
    }
    let ResidualCarry { rows: output, pre } = output;
    let (output, per_layer) = sublayers.per_layer(graph, per_layer, output, weights, constants)?;
    // The n-gram input's windows follow the mixer's bank components, or are
    // the block's bank, over the slot tables it takes as inputs.
    let (output, ngram) = sublayers.ngram(graph, ngram, &mut state_ports, output, weights, constants)?;
    let residual = shape.residual.then(|| output.clone());
    // The decoder's last block ends with its exit: a low-rank exit site,
    // whose partials the readout sums, or the collapse of Sinkhorn streams
    // by the last site's pre or into their mean, which the readout
    // normalizes.
    let carry = ResidualCarry { rows: output, pre };
    let output = match (shape.exit, streams, streams_entries.map(|entries| entries.mixing)) {
        (None, _, _) => carry,
        (Some(ExitTopology::LowRank { epsilon }), Some(streams), Some(MixingEntries::LowRank(sites))) => {
            let (Some(exit), Some(low_rank)) = (sites.exit, streams.low_rank()) else {
                return Err("low-rank streams without their exit site's entries".into());
            };
            let partials = mix_site(
                graph,
                exit,
                planned,
                WeightScope::Target,
                low_rank,
                &carry.rows,
                rows.clone(),
                epsilon.f32(),
                false,
                weights,
            )?
            .partials;
            ResidualCarry::rows(partials)
        }
        (Some(ExitTopology::Collapse(exit)), Some(streams), Some(MixingEntries::Sinkhorn(entries))) => {
            ResidualCarry::rows(collapse(
                graph,
                entries.collapse,
                streams.clone(),
                &carry,
                exit,
                rows.clone(),
                constants,
            )?)
        }
        _ => return Err("the decoder's exit disagrees with its streams' mixing".into()),
    };
    if let Some(taps) = taps.as_mut().filter(|taps| taps.positions.output) {
        taps.ports.output = Some(taps.tap(graph, &output.rows)?);
    }
    let span = begun.map(|begun| (begun, graph.span_mark()));
    Ok(BlockGraphParts {
        residual,
        output,
        state: state_ports,
        taps: taps.map(|taps| taps.ports),
        per_layer,
        ngram,
        sequence,
        ports,
        span,
        operand,
        operand_in,
    })
}

/// A block graph's checked entry bindings, from its program slot: what its
/// topology key holds of its kernels.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct CheckedBlockEntries {
    mixer: block::CheckedMixerEntries,
    feed_forward: Option<block::CheckedFeedForwardEntries>,
    bank: Option<CheckedBankEntries>,
    per_layer: Option<CheckedPerLayerEntries>,
    ngram: Option<CheckedNGramEntries>,
    streams: Option<CheckedStreamsEntries>,
}

impl CheckedBlockEntries {
    fn new(
        state: &StateResourcePlan,
        backend: BackendName,
        slot: TargetBlockProgramSlot,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            mixer: block::CheckedMixerEntries::new(slot.mixer(), backend, slot_lists(state, backend, slot))?,
            feed_forward: slot
                .feed_forward()
                .map(|slot| block::CheckedFeedForwardEntries::new(slot, backend))
                .transpose()?,
            bank: slot.bank().map(|(feed_forward, bank)| CheckedBankEntries::new(feed_forward, bank, backend)).transpose()?,
            per_layer: slot.per_layer().map(CheckedPerLayerEntries::new).transpose()?,
            ngram: slot.ngram().map(CheckedNGramEntries::new),
            streams: slot.block_streams().map(CheckedStreamsEntries::new),
        })
    }
}

/// The metadata graph of a block of `shape` at `numbers` from its checked
/// `entries`, over `planned`: the graph, its constants, and the host heap of
/// the vectors its prepared graph owns (`owned_vectors_heap`).
fn checked_block_graph(
    mut graph: NativeGraphMetadata,
    entries: &CheckedBlockEntries,
    shape: &BlockShape,
    numbers: &BlockNumbers,
    planned: &(impl PlannedWeights + ?Sized),
) -> Result<(NativeGraphMetadata, Vec<GraphConstant>, HostHeap), GraphError> {
    let mut weights = Vec::new();
    let mut constants = Vec::new();
    let rows = &numbers.sublayers.rows;
    let hidden = numbers
        .input
        .clone()
        .map(|carry| {
            carry.try_map(|width| GraphDraft::port_with_class_extent(&mut graph, Element::f32(), &[rows, &width], 0, "M"))
        })
        .transpose()?;
    let tap_elements = [("A", shape.activation)];
    let mut banks = Vec::new();
    let mut parks = Vec::new();
    let bank = layer_bank(shape.stage, entries.bank.as_ref().map(CheckedBankEntries::entries), &mut banks, &mut parks);
    let parts = block_graph(
        &mut graph,
        BlockEntries {
            tap: shape.tap.map(|positions| TapEntry { entry: &tap_elements[..], positions }),
            mixer: entries.mixer.entries()?,
            feed_forward: FeedForwardGraph {
                entries: entries.feed_forward.as_ref().map(block::CheckedFeedForwardEntries::entries),
                bank,
            },
            per_layer: entries.per_layer.as_ref().map(CheckedPerLayerEntries::entries),
            ngram: entries.ngram.as_ref().map(CheckedNGramEntries::entries),
            streams: entries.streams.as_ref().map(CheckedStreamsEntries::entries),
        },
        hidden.as_ref().map(ResidualCarry::as_ref),
        shape,
        numbers,
        planned,
        &mut weights,
        &mut constants,
    )?;
    if let Stage::Run(_) = shape.stage {
        GraphDraft::export(&mut graph, &parts.output.rows)?;
        if let Some(pre) = &parts.output.pre {
            GraphDraft::export(&mut graph, pre)?;
        }
    }
    if let Some(residual) = &parts.residual {
        GraphDraft::export(&mut graph, residual)?;
    }
    let owned = owned_vectors_heap(weights.len(), banks.len(), parks.len(), constants.len(), parts.state_ports());
    Ok((graph, constants, owned))
}

/// A target block graph's topology key: its backend, its checked entries,
/// its shape, and its numbers and weights as the facts they are split into.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct BlockKey {
    backend: BackendName,
    entries: CheckedBlockEntries,
    shape: BlockShape<RealIndex, TableIndex>,
    numbers: BlockNumbers<FactIndex>,
    weights: GraphWeights<FactIndex>,
}

/// What a block graph's recording keeps beside its template: its constants,
/// and the host heap its prepared graph's vectors own, which its topology
/// alone decides.
pub(super) struct RecordedBlock {
    constants: Vec<RecordedConstant>,
    owned: HostHeap,
}

impl TraceOutput for RecordedBlock {
    type Value = (Vec<GraphConstant>, HostHeap);
    fn at(&self, facts: &[u64], reals: &Reals) -> Option<Self::Value> {
        Some((self.constants.at(facts, reals)?, self.owned))
    }
}

/// The target block graphs' traces, one per topology key.
pub(super) static BLOCK_TRACES: NativeGraphTraces<BlockKey, RecordedBlock> = NativeGraphTraces::new();

/// The template of a target block graph of `key` at `facts`.
fn record_block(
    key: &BlockKey,
    facts: &[Dim],
    reals: &Reals,
) -> Result<(NativeGraphResourceTemplate, RecordedBlock), GraphError> {
    let facts = Facts::new(facts, reals);
    let (graph, constants, owned) = checked_block_graph(
        NativeGraphMetadata::new_template(key.backend),
        &key.entries,
        &facts.arguments(key.shape.clone()).tables(|table| facts.table(table)),
        &facts.dims(key.numbers.clone()),
        &facts.dims(key.weights.clone()),
    )?;
    Ok((
        graph.seal_template()?,
        RecordedBlock { constants: constants.iter().map(GraphConstant::recorded).collect(), owned },
    ))
}

/// A class dimension of a block graph's slices, and the number of its split
/// that it varies (`seismic::ClassFact`).
#[derive(Clone, Copy, Debug)]
pub(super) enum ClassNumber {
    Rows,
    Segments,
    Listed,
    Slots,
}

/// The block graph `inputs` names of `slot` on `backend`, with its draft
/// taps (`tap`), certified over each slice set of `slice_sets`, whose class
/// dimensions `classes` names: from the block graphs' traces
/// (`Certified::traced_each`), each with its constants and owned heap.
pub(super) fn certify_block<L: Certified>(
    backend: BackendName,
    slot: TargetBlockProgramSlot,
    tap: Option<(u64, TapPositions)>,
    inputs: BlockGraphInputs<'_>,
    classes: &[(Option<&'static str>, &'static str, ClassNumber)],
    slice_sets: &[Vec<NativeGraphClassSlice>],
) -> Result<Vec<(L, (Vec<GraphConstant>, HostHeap))>, GraphError> {
    with_block_key(backend, slot, tap, inputs, classes, |key, facts, classes| {
        L::traced_each(&BLOCK_TRACES, key, facts, classes, slice_sets, record_block)
    })
}

/// `trace` over the topology key of the block graph `inputs` names of
/// `slot` on `backend`, with its draft taps (`tap`): the key, the facts it
/// is split into, and the facts its classes vary, whose dimensions
/// `classes` names. The key holds the block's checked entries, which live
/// in this frame alone.
fn with_block_key<T>(
    backend: BackendName,
    slot: TargetBlockProgramSlot,
    tap: Option<(u64, TapPositions)>,
    inputs: BlockGraphInputs<'_>,
    classes: &[(Option<&'static str>, &'static str, ClassNumber)],
    trace: impl FnOnce(&BlockKey, &FactSink, &[ClassFact]) -> Result<T, GraphError>,
) -> Result<T, GraphError> {
    let entries = CheckedBlockEntries::new(inputs.state, backend, slot)?;
    let (mut banks, mut parks) = (Vec::new(), Vec::new());
    let bank = layer_bank(inputs.stage, entries.bank.as_ref().map(CheckedBankEntries::entries), &mut banks, &mut parks);
    let mut facts = FactSink::new();
    let (shape, numbers) = inputs.indexed_split(slot, &bank, tap, &mut facts)?;
    // The key names what the block reads relative to it (`relative_scope`),
    // as its graph does.
    let block = inputs.block()?;
    let weights = facts.index(
        GraphWeights::of_block(inputs.load, block, shape.exit.is_some(), shape.carries.carry_out.is_some())?
            .relative_to(block),
    );
    let classes = classes
        .iter()
        .map(|&(scope, dimension, number)| {
            let sublayers = &numbers.sublayers;
            let fact = match number {
                ClassNumber::Rows => sublayers.rows,
                ClassNumber::Segments => sublayers.segments,
                ClassNumber::Listed => sublayers.listed,
                ClassNumber::Slots => sublayers.slots,
            };
            ClassFact { fact: fact.position(), scope, dimension }
        })
        .collect::<Vec<_>>();
    let key = BlockKey { backend, entries, shape, numbers, weights };
    trace(&key, &facts, &classes)
}

/// The metadata graph of block `inputs` names of `slot`, with its draft taps
/// (`tap`), at the model's own numbers over its load: its constants and
/// owned heap beside it.
#[cfg(test)]
pub(super) fn checked_block_graph_draft(
    graph: NativeGraphMetadata,
    slot: TargetBlockProgramSlot,
    tap: Option<(u64, TapPositions)>,
    inputs: BlockGraphInputs<'_>,
) -> Result<(NativeGraphMetadata, Vec<GraphConstant>, HostHeap), GraphError> {
    let entries = CheckedBlockEntries::new(inputs.state, graph.backend(), slot)?;
    let (mut banks, mut parks) = (Vec::new(), Vec::new());
    let bank = layer_bank(inputs.stage, entries.bank.as_ref().map(CheckedBankEntries::entries), &mut banks, &mut parks);
    let (shape, numbers) = inputs.constant_split(slot, &bank, tap, |_| Ok(()))?;
    let planned = inputs.weights()?;
    checked_block_graph(graph, &entries, &shape, &numbers, &planned)
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
        BlockGraphInputs::of_class(load, geometry, state, block_index, staged, class.0, BlockCarries::default()),
    )?;
    Ok((GraphDraft::seal(graph)?, constants))
}
