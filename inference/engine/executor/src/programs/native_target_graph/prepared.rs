use super::{
    GroupExtents, GroupRows,
    block::{BlockCarries, BlockGraphInputs, BlockGraphs, PreparedTargetBlockGraph, SealedBlockGraph, certify_carry},
    bound::{
        BlockBindings, BoundPerLayerEntry, BoundTargetGraphs, GroupBuffers, PassBuffers, TokenRows,
        bind_block_graphs, zeros,
    },
    certify::{TargetFamilyCertificate, certify_target_family},
    classes::{
        BlockGraphEnumeration, BlockListings, BlockOutputScales, BlockServed, Holder, LaunchControlLayout,
        Passes, StagedClass, block_sealed_stages, block_stages, carried_forms, chained_once_runs, class_topology, decode_launch, enumerate_block_graphs, LaunchRuns, once_class,
        slot_selection,
    },
    entry::{
        EntryTokens, PerLayerEntryGraphSource, PreparedPerLayerEntryGraph,
        PreparedTargetEntryGraph, notice_bytes, notice_rows,
    },
    heap::{GraphCounts, PlannedSealedGraphs, SealedGraphsHeap, SealedHeap, block_graph_storage},
};
use crate::{
    AbsorbedQueryShape, AttentionShape, CandidatesBinding, GroupedOutputShape, MixerProgramSlot,
    ModelLoadPlan, ResidentTarget, ResourceLimits, StateResourcePlan, TargetProgramPlan,
    native::AttestedTarget,
    operators,
    operators::{
        BlockKeeps, FeedForward, Mixer, PairedBlock,
        attention::{PickedSource, pooled::PooledSource},
        paired_block, post_norm_epsilon,
        routed::{
            GeneralRoutedShape,
            split::{OnceStage, RunStage, Stage},
        },
        short_conv::ShortConvShape,
        state_space::StateSpaceShape,
        streams::StreamsGeometry,
    },
    programs::{
        graph::tap::{TapEntry, TapPositions},
        native_constants::{ConstantTensors, FamilyConstants, distinct_storage_bytes},
    },
    stored_table::SelectionNotice,
};
use magnitude_family_contracts::{
    ActivationFunction, Decoder, DenseFfn, ExitNorm, GateFunction, HeadNorm, HeadWeights, Indexer,
    IndexerInput, IndexerKeys, NGramGate, NGramMedia, Pool, PoolNorm, RecurrentHeadMapping, Rotary,
    RouteNormalization, RowRounding, SublayerIndex, TapPoint, WeightKind, WeightScope,
};
use magnitude_kernels::selection_notice;
use magnitude_state::{CodecSpec, ComponentSpec, LayerRef};
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
    /// The state each block keeps, which decides the class a launch
    /// selects per block.
    pub(super) keeps: Vec<BlockKeeps>,
    /// Per block, the class axes each `(RowForm, RunStage)` actually reads
    /// (`probe_block_axes`), derived once at certification.
    pub(super) stage_axes: super::classes::StageAxes,
    /// The row-class bound up to which tapped blocks hold dedicated untapped
    /// graphs (`FamilyOutline::untapped_rows`).
    pub(super) untapped_rows: u64,
    /// Per block, its selected read's shape, which decides whether a launch
    /// takes its scoring class.
    selections: Vec<Option<crate::SelectionBinding>>,
    family: NativeGraphFamily,
    max_output_bytes: u64,
    /// Per block, the tap indices its graph writes.
    taps: Vec<BlockTaps>,
    /// Per block, the pooled source its index-list source reads, which its
    /// graph's picked ports bind: a graph its shape shares with other blocks
    /// names none (`PickedSource::pooled`).
    pooled: Vec<Option<PooledSource>>,
    /// The first block of the stateless tail (`stateless_tail`).
    stateless_tail: usize,
    /// The per-layer entry graph per row class, of a per-layer entry.
    pub(super) per_layer_entries: BTreeMap<u64, PreparedPerLayerEntryGraph>,
    /// The rows of each copy of the selection notice a load with stored
    /// tables binds (`notice_rows`).
    notice_rows: Option<u64>,
    /// What its prompt groups' buffers span (`GroupRows`).
    group: GroupRows,
    windows: u64,
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

/// Typed equivalent of `format!("{:?}", (bank, history))` in
/// `block_graph_shapes`: every field derives `PartialEq, Eq` and `Debug` with
/// no floats or skipped fields, so structural equality is identical to the
/// previous Debug-string key.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BlockStateShape {
    bank: Option<Option<Vec<ComponentSpec>>>,
    history: Option<(usize, u32, u32, usize, CodecSpec)>,
}

/// Typed equivalent of `PairedBlock::shape_key` (`block_graph_shapes`):
/// captures the exact same numerical parameters per sublayer, comparing
/// standalone floats by IEEE 754 bit pattern (`to_bits`, matching `Debug`
/// formatting) and hashing vocabulary-long n-gram token maps to a raw 32-byte
/// SHA-256 digest without hex formatting, so equality defines the exact same
/// equivalence classes with zero `format!("{:?}")` allocations.
#[derive(Clone, Debug, PartialEq)]
struct BlockGeometryShape {
    mixer: MixerShapeKey,
    mixer_output_epsilon: Option<u64>,
    feed_forward: Option<(FeedForwardShapeKey, Option<u64>)>,
    per_layer: Option<PerLayerShapeKey>,
    ngram: Option<NGramShapeKey>,
    epsilon: u64,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
enum MixerShapeKey {
    Attention {
        shape: AttentionShape,
        gate_function: i32,
        head_norm_epsilon: u64,
        rotary: Rotary,
        scale: u64,
        selection: Option<SelectionShapeKey>,
        picked: Option<PickedShapeKey>,
        low_rank_query: Option<(u64, u64)>,
        grouped_output: Option<GroupedOutputShape>,
        absorbed_query: Option<AbsorbedQueryShape>,
    },
    GatedDelta {
        convolution_width: u64,
        key_heads: u64,
        value_heads: u64,
        width: u64,
        head_mapping: RecurrentHeadMapping,
        output_gate_function: GateFunction,
        key_norm_epsilon: u64,
        decay_lower: Option<u32>,
    },
    StateSpace {
        shape: StateSpaceShape,
        norm_epsilon: u64,
    },
    ShortConv(ShortConvShape),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct IndexerShapeKey {
    heads: u64,
    width: u64,
    top: u64,
    scale: u64,
    query_rounding: RowRounding,
    has_query_norm: bool,
    has_projected_head_weights: bool,
    candidates: CandidatesBinding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoolShapeKey {
    Mean,
    Gated(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoolNormShapeKey {
    Pooled(u64),
    Token(u64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SelectionShapeKey {
    rate: u64,
    pool: PoolShapeKey,
    norm: PoolNormShapeKey,
    query_input: IndexerInput,
    query_norm_epsilon: Option<u64>,
    indexer: IndexerShapeKey,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PickedShapeKey {
    Own {
        rate: u64,
        gated: bool,
        norm_epsilon: u64,
        rounding: RowRounding,
        latent_keys: (u64, RowRounding),
        indexer: IndexerShapeKey,
    },
    Reindex {
        source: SublayerIndex,
        part: u32,
        indexer: IndexerShapeKey,
    },
    Reused {
        source: SublayerIndex,
        part: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DenseShapeKey {
    intermediate: u64,
    activation: ActivationFunction,
    limit: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FeedForwardShapeKey {
    Dense(DenseShapeKey),
    Routed {
        experts: u64,
        selected: u64,
        intermediate: u64,
        shared_intermediate: Option<u64>,
        normalization: RouteNormalization,
    },
    GeneralRouted(GeneralRoutedShape),
    Parallel {
        dense: DenseShapeKey,
        routed: GeneralRoutedShape,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PerLayerShapeKey {
    layer: u64,
    width: u64,
    post_norm_epsilon: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NGramShapeKey {
    context: u64,
    heads_per_order: u64,
    width: u64,
    mixing: Option<(u64, u64)>,
    gate_is_norms: bool,
    epsilon: u64,
    multipliers: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    token_map_digest: Option<[u8; 32]>,
    fill: u64,
    boundary: Option<u64>,
    media: NGramMedia,
}

/// Inline-small tensor shape dimensions avoiding a heap `Vec<u64>` allocation
/// per weight in [`BlockGraphShape`] (all standard weight tensors have rank <= 4).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum WeightDims {
    Inline { len: u8, dims: [u64; 4] },
    Heap(Box<[u64]>),
}

impl WeightDims {
    #[inline]
    fn from_slice(slice: &[u64]) -> Self {
        if slice.len() <= 4 {
            let mut dims = [0u64; 4];
            dims[..slice.len()].copy_from_slice(slice);
            Self::Inline { len: slice.len() as u8, dims }
        } else {
            Self::Heap(slice.into())
        }
    }
}

/// The subset of [`BlockGraphShape`] that determines a block's
/// `OnceStage::Pass` and `OnceStage::Fill` graphs: its feed-forward sublayer
/// geometry, its sublayer-1 weights, and its expert-bank binding. Blocks that
/// differ only in their mixer, taps, residual export, state, or carries share
/// identical `OnceGraphShape`s.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct OnceGraphShape {
    banked: Option<crate::BankBinding>,
    epsilon: u64,
    feed_forward: Option<(FeedForwardShapeKey, Option<u64>)>,
    weights: Vec<((u32, Option<u32>), WeightKind, Element, WeightDims, u64)>,
}

/// The inputs that determine a block's sealed graph apart from its layer
/// index: its geometry, the resident element and shape of each weight role
/// within the block, and its recurrent state components. Blocks with equal
/// shapes share plans.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct BlockGraphShape {
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
    /// The block offset back to the streamed layer whose route predicts
    /// this block's (whose hint ring its route reads, `StreamedLayer::hinted`);
    /// none for a block that streams none or the first streamed one. A
    /// graph shared by blocks binds each one's predictor at the same offset.
    hinted_by: Option<u32>,
    /// The operand carries its decode graphs declare (`block_carries`):
    /// the first block's reads none, the last block's writes none, so
    /// neither shares an interior block's plans. Each names its reader's
    /// mixer binding, whose kernels decide the forms a sealed graph
    /// carries it in (`CarrySide`): blocks share plans only over equal
    /// readers, so over equal forms, whatever a load's tuning chose.
    carries: BlockCarries<MixerProgramSlot>,
    /// A streamed routed layer's bank binding.
    banked: Option<crate::BankBinding>,
    /// (sublayer, branch) of each role, its resident element, shape and
    /// accumulator-scale port extent.
    weights: Vec<((u32, Option<u32>), WeightKind, Element, WeightDims, u64)>,
    /// The layers a streamed routed layer's route predicts (its lead): each
    /// one's block offset from this block and the weights the route reads
    /// of it (`BankLayer::hinted`). A graph shared by blocks binds each
    /// one's layers at the same offsets (`block_scope`).
    predicts: Vec<(u32, Vec<crate::planning::HintWeight>)>,
    state: BlockStateShape,
    geometry: BlockGeometryShape,
}

impl BlockGraphShape {
    /// The shape without its drafter's writes: no taps, no exported
    /// residual (`FamilyOutline::untapped`), its carries the target's own,
    /// as a tapped block's untapped graphs run in its load.
    pub(super) fn undrafted(&self) -> Self {
        Self { tapped: TapPositions::default(), residual: false, ..self.clone() }
    }

    /// The shape with the operand carries `carries` (`block_carries`).
    pub(super) fn carrying(self, carries: BlockCarries<MixerProgramSlot>) -> Self {
        Self { carries, ..self }
    }

    /// The feed-forward-only key that governs `OnceStage::Pass` and
    /// `OnceStage::Fill` graphs.
    pub(super) fn once_shape(&self) -> OnceGraphShape {
        OnceGraphShape {
            banked: self.banked,
            epsilon: self.geometry.epsilon,
            feed_forward: self.geometry.feed_forward,
            weights: self
                .weights
                .iter()
                .filter(|((sublayer, _), ..)| *sublayer == 1)
                .cloned()
                .collect(),
        }
    }

    /// Fast collision-safe fingerprint consistent with `PartialEq` (`a == b =>
    /// a.fx_hash() == b.fx_hash()`), hashing only `Eq + Hash` fields for
    /// `O(1)` bucketed lookups followed by exact `==` checks.
    pub(super) fn fx_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = super::certify::FxHasher::default();
        self.widens_streams.hash(&mut h);
        self.mixes_exit.hash(&mut h);
        (self.tapped.entry, self.tapped.middle, self.tapped.output).hash(&mut h);
        self.residual.hash(&mut h);
        self.sets_step.hash(&mut h);
        self.hinted_by.hash(&mut h);
        (self.carries.carry_in.is_some(), self.carries.carry_out.is_some()).hash(&mut h);
        self.banked.is_some().hash(&mut h);
        self.weights.hash(&mut h);
        self.predicts.len().hash(&mut h);
        for (offset, hints) in &self.predicts {
            offset.hash(&mut h);
            hints.len().hash(&mut h);
        }
        self.state.history.hash(&mut h);
        self.state.bank.as_ref().map(|b| b.as_ref().map(Vec::len)).hash(&mut h);
        std::mem::discriminant(&self.geometry.mixer).hash(&mut h);
        self.geometry.mixer_output_epsilon.hash(&mut h);
        self.geometry
            .feed_forward
            .as_ref()
            .map(|(ff, eps)| (std::mem::discriminant(ff), *eps))
            .hash(&mut h);
        self.geometry
            .per_layer
            .as_ref()
            .map(|p| (p.layer, p.width, p.post_norm_epsilon))
            .hash(&mut h);
        self.geometry
            .ngram
            .as_ref()
            .map(|n| (n.context, n.width, n.token_map_digest))
            .hash(&mut h);
        self.geometry.epsilon.hash(&mut h);
        h.finish()
    }
}

impl OnceGraphShape {
    /// Fast collision-safe fingerprint consistent with `PartialEq`.
    pub(super) fn fx_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = super::certify::FxHasher::default();
        self.banked.is_some().hash(&mut h);
        self.epsilon.hash(&mut h);
        self.feed_forward
            .as_ref()
            .map(|(ff, eps)| (std::mem::discriminant(ff), *eps))
            .hash(&mut h);
        self.weights.hash(&mut h);
        h.finish()
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

/// Each block's pooled source (`PickedSource::pooled`): the attention it
/// opens with, as its graph's (`BlockSublayers::picked`).
fn pooled_sources(geometry: &Decoder) -> Result<Vec<Option<PooledSource>>, String> {
    (0..geometry.blocks.len())
        .map(|index| {
            let block = u32::try_from(index).map_err(|_| "block index exceeds u32")?;
            let Some(attention) = geometry.attention(SublayerIndex { block, sublayer: 0 }) else {
                return Ok(None);
            };
            Ok(operators::attention::picked_source(attention)
                .map_err(|error| error.to_string())?
                .map(|source| source.pooled(block)))
        })
        .collect()
}

/// The first block of the longest run of trailing blocks that publish no
/// state: blocks keeping a history and no bank, that read another layer's
/// history (a Shared domain, which they never append to), own no pooled
/// source rows or indexer keys (which they would append) and write no draft
/// taps. A row's output from such a block reaches nothing but later blocks
/// of the run and the readout, so a step need run the run only over the
/// rows its readout consumes. The block count when the last block publishes state.
pub(super) fn stateless_tail(state: &StateResourcePlan, keeps: &[BlockKeeps], taps: &[BlockTaps]) -> usize {
    trailing_run(keeps.len(), |index| {
        let block = index as u32;
        let layer = LayerRef::Target(block);
        let target = state.target_state();
        keeps[index] == BlockKeeps::History
            && taps[index] == BlockTaps::default()
            && target
                .layer_history(layer)
                .is_some_and(|history| history.component.layer != layer)
            && !target.domains.iter().any(|store| {
                store.components.iter().any(|component| {
                    matches!(
                        component.layer,
                        LayerRef::Pooled { block: owner, .. } | LayerRef::IndexKeys { block: owner, .. }
                            if owner == block
                    )
                })
            })
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

/// A target sublayer (or branch) scope a block graph names relative to its
/// block (`relative_scope`), bound for block `to`: the same sublayer (or
/// branch) at the same offset from the block, so a scope of a later block
/// (a streamed route's layers ahead) moves with it. A decoder-wide scope
/// (the exit site the last block ends with) is every block's own.
pub(super) fn block_scope(scope: WeightScope, to: usize) -> Result<WeightScope, String> {
    // An offset is a wrapped `u32` (`relative_scope`): earlier blocks wrap
    // below zero.
    let moved = |block: u32| {
        i64::try_from(to)
            .ok()
            .map(|to| to + i64::from(block as i32))
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
fn predicted_layers_for(
    bank: Option<&crate::ExpertBankPlan>,
    layer: Option<usize>,
    index: usize,
) -> Result<Vec<(u32, Vec<crate::planning::HintWeight>)>, String> {
    let (Some(bank), Some(layer)) = (bank, layer) else {
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
fn hinted_by_for(
    bank: Option<&crate::ExpertBankPlan>,
    layer: Option<usize>,
    index: usize,
) -> Result<Option<u32>, String> {
    let (Some(bank), Some(layer)) = (bank, layer) else {
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
fn banked_layer_for(
    bank: Option<&crate::ExpertBankPlan>,
    layer: Option<usize>,
) -> Result<Option<crate::BankBinding>, String> {
    bank.zip(layer)
        .map(|(bank, layer)| bank.binding(layer))
        .transpose()
}

#[inline]
fn weight_kind_order_hash(kind: WeightKind) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = super::certify::FxHasher::default();
    kind.hash(&mut h);
    h.finish()
}

/// Stack-allocated `Debug` buffer used to sort `WeightKind` in the exact same
/// lexicographical order as `format!("{kind:?}")` without heap allocations.
#[derive(Eq, PartialEq)]
struct StackDebug {
    bytes: [u8; 64],
    len: usize,
}

impl StackDebug {
    fn of(value: &impl std::fmt::Debug) -> Self {
        use std::fmt::Write;
        let mut out = Self { bytes: [0; 64], len: 0 };
        let _ = write!(out, "{value:?}");
        out
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl std::fmt::Write for StackDebug {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let take = s.len().min(self.bytes.len() - self.len);
        self.bytes[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

impl Ord for StackDebug {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl PartialOrd for StackDebug {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn dense_shape_key(dense: &DenseFfn) -> DenseShapeKey {
    DenseShapeKey {
        intermediate: dense.intermediate,
        activation: dense.up.activation(),
        limit: dense.up.limit().map(|limit| f64::from(limit).to_bits()),
    }
}

fn indexer_shape_key(
    site: Option<operators::attention::DecoderSite<'_>>,
    part: u64,
    indexer: &Indexer,
) -> Result<IndexerShapeKey, String> {
    Ok(IndexerShapeKey {
        heads: indexer.heads,
        width: indexer.width,
        top: indexer.top,
        scale: indexer.scale.to_bits(),
        query_rounding: indexer.query.rounding,
        has_query_norm: matches!(indexer.query.norm, HeadNorm::Rms(_)),
        has_projected_head_weights: matches!(indexer.head_weights, HeadWeights::Projected(_)),
        candidates: operators::attention::candidates(site, part, indexer)
            .map_err(|error| error.to_string())?,
    })
}

fn token_map_digest<'a>(
    map: &'a [u32],
    cached: &mut Option<(&'a [u32], [u8; 32])>,
) -> [u8; 32] {
    if let Some((prev, digest)) = cached {
        if *prev == map {
            return *digest;
        }
    }
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64];
    let mut chunks = map.chunks_exact(16);
    for chunk in &mut chunks {
        for (dst, &id) in buf.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&id.to_le_bytes());
        }
        hasher.update(buf);
    }
    for &id in chunks.remainder() {
        hasher.update(id.to_le_bytes());
    }
    let digest: [u8; 32] = hasher.finalize().into();
    *cached = Some((map, digest));
    digest
}

fn block_geometry_shape<'a>(
    paired: &PairedBlock<'a>,
    site: Option<operators::attention::DecoderSite<'_>>,
    token_map_cache: &mut Option<(&'a [u32], [u8; 32])>,
) -> Result<BlockGeometryShape, String> {
    let mixer = match paired.mixer {
        Mixer::Attention(operator) => {
            let mut shape = operators::attention::shape(0, operator, site)
                .map_err(|error| error.to_string())?;
            shape.widths.picks = 0;
            let selection = operators::attention::selected_read(&operator.fresh().reads)
                .map_err(|error| error.to_string())?
                .map(|read| -> Result<SelectionShapeKey, String> {
                    let norm = match &read.keys.norm {
                        PoolNorm::Pooled(norm) => PoolNormShapeKey::Pooled(norm.epsilon.to_bits()),
                        PoolNorm::Token { norm, .. } => {
                            PoolNormShapeKey::Token(norm.epsilon.to_bits())
                        }
                    };
                    let pool = match &read.keys.pool {
                        Pool::Mean => PoolShapeKey::Mean,
                        Pool::SoftmaxGated { position, .. } => {
                            PoolShapeKey::Gated(position.is_some())
                        }
                        Pool::Identity => {
                            unreachable!("selected_read admits pooling of two rows or more")
                        }
                    };
                    let query_norm_epsilon = match &read.indexer.query.norm {
                        HeadNorm::Rms(norm) => Some(norm.epsilon.to_bits()),
                        HeadNorm::None => None,
                    };
                    Ok(SelectionShapeKey {
                        rate: read.keys.rate,
                        pool,
                        norm,
                        query_input: read.indexer.query.input,
                        query_norm_epsilon,
                        indexer: indexer_shape_key(None, 0, read.indexer)?,
                    })
                })
                .transpose()?;
            let picked = operators::attention::picked_source(operator)
                .map_err(|error| error.to_string())?
                .map(|source| -> Result<PickedShapeKey, String> {
                    Ok(match source {
                        PickedSource::Own { part, rows, indexer } => PickedShapeKey::Own {
                            rate: rows.rate,
                            gated: operators::attention::pooled::gated(rows),
                            norm_epsilon: operators::attention::pooled_norm(rows)
                                .map_err(|error| error.to_string())?
                                .epsilon
                                .to_bits(),
                            rounding: rows.rounding,
                            latent_keys: match &indexer.keys {
                                IndexerKeys::Latent { norm, rounding, .. } => {
                                    (norm.epsilon.to_bits(), *rounding)
                                }
                                IndexerKeys::Pooled(_) | IndexerKeys::Shared { .. } => {
                                    unreachable!(
                                        "picked_source admits an own source's latent keys alone"
                                    )
                                }
                            },
                            indexer: indexer_shape_key(site, u64::from(part), indexer)?,
                        },
                        PickedSource::Reindex { source, part, indexer } => {
                            PickedShapeKey::Reindex {
                                source,
                                part,
                                indexer: indexer_shape_key(site, u64::from(part), indexer)?,
                            }
                        }
                        PickedSource::Reused { source, part, .. } => {
                            PickedShapeKey::Reused { source, part }
                        }
                    })
                })
                .transpose()?;
            MixerShapeKey::Attention {
                shape,
                gate_function: operators::attention::gate_function(operator),
                head_norm_epsilon: operators::attention::head_norm_epsilon(
                    operator,
                    paired.epsilon(),
                )
                .map_err(|error| error.to_string())?
                .to_bits(),
                rotary: operator.rotary.clone(),
                scale: operator.scale.to_bits(),
                selection,
                picked,
                low_rank_query: operators::attention::low_rank_query(operator)
                    .map(|(rank, epsilon)| (rank, epsilon.to_bits())),
                grouped_output: operators::attention::grouped_output(operator),
                absorbed_query: operators::attention::absorbed_query(operator),
            }
        }
        Mixer::GatedDelta(admitted) => {
            let operator = admitted.operator();
            MixerShapeKey::GatedDelta {
                convolution_width: operator.convolution_width,
                key_heads: operator.key_heads,
                value_heads: operator.value_heads,
                width: operator.width,
                head_mapping: operator.head_mapping,
                output_gate_function: operator.output_gate.function,
                key_norm_epsilon: operator.key_norm_epsilon.to_bits(),
                decay_lower: admitted.decay_lower().map(f32::to_bits),
            }
        }
        Mixer::StateSpace(operator) => MixerShapeKey::StateSpace {
            shape: StateSpaceShape::of(0, operator).map_err(|error| error.to_string())?,
            norm_epsilon: operator.norm.epsilon.to_bits(),
        },
        Mixer::ShortConv(operator) => MixerShapeKey::ShortConv(
            ShortConvShape::of(0, operator).map_err(|error| error.to_string())?,
        ),
    };
    let feed_forward = paired
        .feed_forward
        .map(|feed_forward| -> Result<(FeedForwardShapeKey, Option<u64>), String> {
            let op = match feed_forward.op {
                FeedForward::Dense(operator) => {
                    FeedForwardShapeKey::Dense(dense_shape_key(operator))
                }
                FeedForward::Routed(operator) => FeedForwardShapeKey::Routed {
                    experts: operator.experts,
                    selected: operator.selected,
                    intermediate: operator.intermediate,
                    shared_intermediate: operator.shared.as_ref().map(|shared| shared.intermediate),
                    normalization: operator.router.normalization,
                },
                FeedForward::GeneralRouted(routed) => FeedForwardShapeKey::GeneralRouted(
                    GeneralRoutedShape::of(0, routed).map_err(|error| error.to_string())?,
                ),
                FeedForward::Parallel(branches) => FeedForwardShapeKey::Parallel {
                    dense: dense_shape_key(branches.dense),
                    routed: GeneralRoutedShape::of(0, branches.routed)
                        .map_err(|error| error.to_string())?,
                },
            };
            Ok((op, post_norm_epsilon(feed_forward.output).map(f64::to_bits)))
        })
        .transpose()?;
    let per_layer = paired.per_layer.map(|sublayer| PerLayerShapeKey {
        layer: sublayer.op.layer,
        width: sublayer.op.width,
        post_norm_epsilon: post_norm_epsilon(sublayer.output).map(f64::to_bits),
    });
    let ngram = paired.ngram.map(|sublayer| {
        let op = sublayer.op;
        let hash = &op.hash;
        NGramShapeKey {
            context: hash.context,
            heads_per_order: hash.heads_per_order,
            width: op.width,
            mixing: op
                .mixing
                .as_ref()
                .map(|mixing| (mixing.kernel, mixing.dilation)),
            gate_is_norms: matches!(op.gate, NGramGate::Norms { .. }),
            epsilon: op.epsilon.to_bits(),
            multipliers: hash.multipliers.clone(),
            sizes: hash.sizes.clone(),
            offsets: hash.offsets.clone(),
            token_map_digest: hash
                .token_map
                .as_deref()
                .map(|map| token_map_digest(map, token_map_cache)),
            fill: hash.fill,
            boundary: hash.boundary,
            media: hash.media,
        }
    });
    Ok(BlockGeometryShape {
        mixer,
        mixer_output_epsilon: post_norm_epsilon(paired.mixer_output).map(f64::to_bits),
        feed_forward,
        per_layer,
        ngram,
        epsilon: paired.epsilon().to_bits(),
    })
}

pub(super) fn block_graph_shapes(
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    tapped: &[BlockTaps],
    carries: &[BlockCarries<MixerProgramSlot>],
    residual: Option<usize>,
) -> Result<Vec<BlockGraphShape>, String> {
    let block_count = geometry.blocks.len();
    let components = &state.target_state().recurrent_components;
    let est_per_block = (load.weights().size_hint().0 / block_count.max(1)).max(8);
    let mut block_weights: Vec<Vec<((u32, Option<u32>), WeightKind, Element, WeightDims, u64)>> =
        (0..block_count).map(|_| Vec::with_capacity(est_per_block)).collect();
    for weight in load.weights() {
        let (block, sublayer) = match weight.role.scope {
            WeightScope::TargetSublayer(sublayer) => {
                (sublayer.block as usize, (sublayer.sublayer, None))
            }
            WeightScope::TargetBranch { sublayer, branch } => {
                (sublayer.block as usize, (sublayer.sublayer, Some(branch)))
            }
            _ => continue,
        };
        if let Some(bucket) = block_weights.get_mut(block) {
            bucket.push((
                sublayer,
                weight.role.kind,
                weight.resident,
                WeightDims::from_slice(&weight.shape),
                weight.scale_extent(),
            ));
        }
    }
    let bank = load.expert_bank();
    let bank_layers = super::classes::bank_layer_by_block(load, block_count);
    let mut bank_cursor = 0usize;
    let mut token_map_cache: Option<(&[u32], [u8; 32])> = None;
    let mut shapes = Vec::with_capacity(block_count);
    for (index, (block, mut weights)) in geometry.blocks.iter().zip(block_weights).enumerate() {
        let paired = paired_block(block).map_err(|error| error.to_string())?;
        weights.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| weight_kind_order_hash(a.1).cmp(&weight_kind_order_hash(b.1)))
                .then_with(|| {
                    if a.1 == b.1 {
                        std::cmp::Ordering::Equal
                    } else {
                        StackDebug::of(&a.1).cmp(&StackDebug::of(&b.1))
                    }
                })
        });
        // Blocks share a plan only over equal bank components, and
        // within one history domain's geometry and codec.
        let keeps = paired.keeps();
        let bank_count = paired.bank_components();
        let bank_state = match keeps.bank() {
            false => None,
            true => Some(
                components
                    .get(bank_cursor..bank_cursor + bank_count)
                    .map(<[ComponentSpec]>::to_vec),
            ),
        };
        bank_cursor += bank_count;
        let block_u32 = u32::try_from(index).map_err(|_| "block index exceeds u32")?;
        let history = match keeps.history() {
            false => None,
            true => {
                let history = state
                    .target_state()
                    .layer_history(LayerRef::Target(block_u32))
                    .ok_or("attention block has no history domain")?;
                Some((
                    history.store.rows,
                    history.store.slab_rows,
                    history.store.page_rows,
                    history.store.span_limit,
                    history.component.codec,
                ))
            }
        };
        let state = BlockStateShape { bank: bank_state, history };
        let geometry_shape = block_geometry_shape(
            &paired,
            Some(operators::attention::DecoderSite {
                decoder: geometry,
                index: SublayerIndex {
                    block: block_u32,
                    sublayer: 0,
                },
            }),
            &mut token_map_cache,
        )?;
        let bank_layer = bank_layers[index];
        shapes.push(BlockGraphShape {
            geometry: geometry_shape,
            weights,
            state,
            widens_streams: index == 0
                && StreamsGeometry::of(geometry)
                    .map_err(|error| error.to_string())?
                    .is_some(),
            mixes_exit: index + 1 == block_count
                && matches!(geometry.exit.norm, ExitNorm::HyperConnectionHead(_)),
            tapped: tapped[index].positions(),
            residual: residual == Some(index),
            sets_step: bank.zip(bank_layer).is_some_and(|(bank, layer)| bank.sets_step(layer)),
            predicts: predicted_layers_for(bank, bank_layer, index)?,
            banked: banked_layer_for(bank, bank_layer)?,
            hinted_by: hinted_by_for(bank, bank_layer, index)?,
            carries: carries[index],
        });
    }
    Ok(shapes)
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
        let _sizing = crate::programs::graph::sizing::SizingScope::enter_with_device(Some(
            *device.properties(),
        ));
        let certificate =
            certify_target_family::<seismic::NativeGraphLayout>(
                device.backend(),
                load,
                geometry,
                state,
                plan,
                limits,
                pass,
            )
            .map_err(|error| error.to_string())?;
        let TargetFamilyCertificate { outline, served, passes, group: certificate_group, .. } = certificate;
        let largest_rows = outline.largest_rows;
        let tap_width = outline.tap_width;
        let stateless_tail = outline.stateless_tail;
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
        let stage_axes = certificate.stage_axes;
        let BlockGraphEnumeration {
            sealing,
            sharing,
            listings,
            controls,
            launch_classes,
            untapped_sealing,
            untapped_sharing,
            exact: _,
        } = enumerate_block_graphs(
            &outline,
            &served,
            passes,
            state,
            plan,
            BlockOutputScales::Read(&scales),
            |index, form, stage| stage_axes.of(index, form, stage),
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
        let notice = match load.copies_token_rows() {
            true => Some(notice.ok_or("a load copying token rows without the copy entry")?),
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
        // A tapped block's untapped graphs (`untapped`) are sealed as the
        // block without its drafter's writes, into their own layouts.
        let certified_of = |index: usize, staged: StagedClass, untapped: bool| {
            let layouts = match untapped {
                true => &certificate.untapped[index],
                false => &certificate.blocks[index],
            };
            layouts.get(&class_topology(staged)).ok_or_else(|| {
                format!("target graph class {staged:?} of block {index} (untapped: {untapped}) has no certified layout")
            })
        };
        let sealing = sealing
            .into_iter()
            .map(|(index, staged)| (index, staged, false))
            .chain(untapped_sealing.into_iter().map(|(index, staged)| (index, staged, true)))
            .collect::<Vec<_>>();
        let seal_block = |&(index, staged, untapped): &(usize, StagedClass, bool)| {
            let ((rows, segments, slots, listed, scored), stage) = staged;
            let began = Instant::now();
            let positions = match untapped {
                true => TapPositions::default(),
                false => outline.tapped[index].positions(),
            };
            let tap = match (positions.any(), tap_width, &handles.taps) {
                (false, ..) => None,
                (true, Some(width), Some(taps)) => Some((width, TapEntry { entry: &taps.tap, positions })),
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
                    residual: !untapped && outline.exports_residual(index),
                    ..BlockGraphInputs::of_class(load, geometry, state, index, staged, capacity, outline.carries[index])
                },
                |reader| carried_forms(handles, plan, index + reader as usize),
                &certified_of(index, staged, untapped)?.layout,
                &constants,
            )
            .map_err(|error| {
                format!(
                    "target graph row class {rows}, history segments {segments}, request slots {slots}, listed {listed}, scored {scored}, {stage:?} of block {index} (untapped: {untapped}): {error}"
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
        for &(index, (_, stage), untapped) in &sealing {
            match untapped {
                true => class_counts[index].untapped += 1,
                false => class_counts[index].count(stage),
            }
            sealed_counts.count(stage);
        }
        for &(index, (_, stage), _) in &sharing {
            class_counts[index].count(stage);
        }
        for &(index, ..) in &untapped_sharing {
            class_counts[index].untapped += 1;
        }
        let mut blocks_by_class = class_counts
            .iter()
            .map(|counts| BlockGraphs {
                runs: Vec::with_capacity(counts.runs),
                passes: Vec::with_capacity(counts.passes),
                untapped: Vec::with_capacity(counts.untapped),
            })
            .collect::<Vec<_>>();
        for (&(index, class, untapped), block) in sealing.iter().zip(sealed) {
            let block = block?;
            sealed_graphs += 1;
            let certified = certified_of(index, class, untapped)?;
            let graph = || format!("target graph class {class:?} of block {index} (untapped: {untapped})");
            let graphs = &mut blocks_by_class[index];
            let plan = match block {
                SealedBlockGraph::Run(block) => {
                    held.own(&block, certified, graph)?;
                    let plan = block.plan.clone();
                    match untapped {
                        true => graphs.untapped.push((class, block)),
                        false => graphs.runs.push((class, block)),
                    }
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
        // A tapped block's untapped graph another serves holds its holder's
        // prepared graph: a block's own of its shape, or an earlier tapped
        // block's untapped one, which ascend by block.
        for (index, class, holder) in untapped_sharing {
            let graphs = match holder {
                Holder::Block(block) => &blocks_by_class[block].runs,
                Holder::Untapped(block) => &blocks_by_class[block].untapped,
            };
            let at = graphs
                .binary_search_by(|(sealed, _)| sealed.cmp(&class))
                .map_err(|_| format!("{holder:?} holds no graph of class {class:?}"))?;
            let graph = graphs[at].1.clone();
            blocks_by_class[index].untapped.push((class, graph));
        }
        // Every decode boundary's graphs agree on its carry, in each form a
        // step runs either side in (tapped or untapped, whole or
        // resident): none into the first block, none out of the last.
        let decode_runs = |graphs: &BlockGraphs| {
            graphs
                .runs
                .iter()
                .chain(&graphs.untapped)
                .filter(|((class, stage), _)| class.0 == 1 && matches!(stage, Stage::Run(run) if run.place() == RunStage::Whole))
                .map(|(_, graph)| Arc::clone(graph))
                .collect::<Vec<_>>()
        };
        let decodes = blocks_by_class.iter().map(decode_runs).collect::<Vec<_>>();
        let unpaired = |graphs: &[Arc<PreparedTargetBlockGraph>], side: fn(&PreparedTargetBlockGraph) -> bool| {
            graphs.iter().any(|graph| side(graph))
        };
        if decodes.first().is_some_and(|first| unpaired(first, |graph| graph.operand_in.is_some()))
            || decodes.last().is_some_and(|last| unpaired(last, |graph| graph.operand.is_some()))
        {
            return Err("an operand carried into the first block or out of the last".into());
        }
        for (index, pair) in decodes.windows(2).enumerate() {
            for from in &pair[0] {
                for to in &pair[1] {
                    certify_carry(index, from, to)?;
                }
            }
        }
        let has_carry = decodes.iter().flatten().any(|graph| graph.operand.is_some());
        assert_eq!(
            has_carry,
            certificate_group.operand_carry.is_some(),
            "invariant violated: sealed decode graphs' operand carries disagree with GroupRows"
        );
        let storage = block_graph_storage(&class_counts, sealed_counts)?;
        let held_storage = block_graph_storage(
            &blocks_by_class
                .iter()
                .map(|graphs| GraphCounts {
                    runs: graphs.runs.capacity(),
                    passes: graphs.passes.capacity(),
                    untapped: graphs.untapped.capacity(),
                })
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
        let windows = load.expert_bank().map_or(1, crate::ExpertBankPlan::windows);
        let launch_runs = LaunchRuns::of(
            &served,
            |index, form, stage| stage_axes.of(index, form, stage),
            blocks_by_class.iter().enumerate().flat_map(|(index, graphs)| {
                // A launch of a class a tapped block's untapped graphs serve
                // runs either form of the block, as its rows demand
                // features or not: it counts both.
                let runs = graphs.runs.iter().chain(&graphs.untapped).map(move |(staged, block)| (index, *staged, &block.plan));
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
                    .chain(launch_runs.of_class(class, passes, windows).copied())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // What a chain of each length beyond one launch further runs of
        // the graphs every split block runs once a group
        // (`chained_once_runs`), as the plan certified them.
        let chains = (2..=limits.chain_capacity.get())
            .map(|launches| {
                let further = chained_once_runs(passes, launches, windows);
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
            bytes(sealed.pinned),
            bytes(sealed.union),
            bytes(bound.pinned),
            bytes(bound.union),
        );
        assert!(
            sealed.within(&bound),
            "invariant violated: the target family's sealed graphs, {sealed:?}, exceed what the planner charged, {bound:?}"
        );
        crate::programs::graph::assert_storage_eq("target", &family, certificate.resources.storage);
        assert_eq!(
            max_output_bytes,
            certificate.resources.storage.output,
            "invariant violated: target family max_output_bytes differs from certified output bytes"
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
            windows,
            upload_regions: certificate.upload_regions,
            max_segments: outline.max_segments,
            keeps: outline.keeps,
            stage_axes,
            untapped_rows: outline.untapped_rows,
            selections: plan.blocks().iter().copied().map(slot_selection).collect(),
            family,
            max_output_bytes,
            taps: outline.tapped,
            pooled: pooled_sources(geometry)?,
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

    pub(crate) fn windows(&self) -> u64 {
        self.windows
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

    /// Block `index`'s selected read, when it reads one.
    pub(crate) fn selection(&self, index: usize) -> Option<crate::SelectionBinding> {
        self.selections.get(index).copied().flatten()
    }

    /// The pooled source block `index`'s index-list source reads, if it has
    /// one: what its graph's picked ports bind.
    pub(crate) fn pooled_source(&self, index: usize) -> Option<PooledSource> {
        self.pooled.get(index).copied().flatten()
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

    /// The pass budget the load holds: the launches a group's pass covers,
    /// the plan's within the arena's cap (`group_pass_rows`), whose places
    /// the arena's prompt-group class holds.
    pub fn held_pass(&self) -> crate::PassLaunches {
        crate::PassLaunches::new(std::num::NonZeroUsize::new(self.passes.launches()).expect("a pass covers a launch"))
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
        charges: &seismic::Charges,
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
            charges,
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
                    .bind_static(charges, &fixed)
                    .map_err(|error| format!("target embedding graph class {key:?}: {error}"))?,
            );
        }
        let mut constants = ConstantTensors::new(charges);
        let mut bound = Vec::with_capacity(self.blocks_by_class.len());
        if resident.blocks != self.blocks_by_class.len() {
            return Err("resident target blocks disagree with the prepared graphs".into());
        }
        for (index, graphs) in self.blocks_by_class.iter().enumerate() {
            // The block's classes ascend in its vectors, so do its bindings.
            bound.push(BlockBindings {
                runs: bind_block_graphs(&graphs.runs, index, resident, charges, &mut constants)?,
                passes: bind_block_graphs(&graphs.passes, index, resident, charges, &mut constants)?,
                untapped: bind_block_graphs(&graphs.untapped, index, resident, charges, &mut constants)?,
            });
        }
        // The pass's merged experts and row table, which every group's passes
        // write and its backs read; the row table holds every first row and
        // row count, entry `i` holding `i`.
        let pass = match super::SplitLayers::of(&self.served) {
            None => None,
            Some(layers) => {
                let split = layers.at_rows(self.region_start(1));
                let (element, extents) = &split.merged.table;
                let counts = (0..i32::try_from(extents[0]).map_err(|_| "a pass's rows exceed i32")?)
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>();
                Some(PassBuffers {
                    experts: zeros(charges, &split.merged.experts).map_err(|error| error.to_string())?,
                    table: seismic::Tensor::from_host(charges, *element, extents, &counts)
                        .map_err(|error| format!("merged row table allocation failed: {error}"))?,
                    layers,
                })
            }
        };
        let floor = Rc::new(GroupBuffers::new(self, charges, 1, pass.as_ref()).map_err(|error| error.to_string())?);
        let per_layer = match (&resident.per_layer, self.group.per_layer) {
            (None, None) => None,
            (Some(weights), Some(extents)) => Some(BoundPerLayerEntry {
                bound: self
                    .per_layer_entries
                    .iter()
                    .map(|(rows, graph)| {
                        let absent = seismic::Tensor::from_host(
                            charges,
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
                            .bind_static(charges, &fixed)
                            .map(|bound| (*rows, bound))
                            .map_err(|error| format!("per-layer entry graph rows {rows}: {error}"))
                    })
                    .collect::<Result<_, String>>()?,
                // The rows a single-row step leaves unwritten in the floor's
                // one region hold tokens' rows.
                token_rows: TokenRows::new(weights.table.rows(), extents[0] as usize - 1),
                table: weights.table.clone(),
            }),
            _ => return Err("resident per-layer weights disagree with the prepared graphs".into()),
        };
        let stored = resident.sublayers.stored_blocks()?;
        let notice = self
            .notice_rows
            .map(|rows| SelectionNotice::new(charges, rows))
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

#[cfg(test)]
mod state_shape_tests {
    use super::BlockStateShape;
    use magnitude_state::{Codec, CodecSpec, HistoryValue};
    use seismic::DType;

    #[test]
    fn block_state_shape_distinguishes_page_rows_span_limit_and_codec() {
        let bf16 = CodecSpec {
            key: Codec::Dense { dtype: DType::BF16 },
            key_width: 128,
            value: HistoryValue::Own {
                codec: Codec::Dense { dtype: DType::BF16 },
                width: 128,
            },
            packing: 1,
        };
        let f16 = CodecSpec {
            key: Codec::Dense { dtype: DType::F16 },
            ..bf16
        };
        let base = BlockStateShape {
            bank: None,
            history: Some((4096, 1024, 256, 18, bf16)),
        };
        let different_page_rows = BlockStateShape {
            bank: None,
            history: Some((4096, 1024, 512, 18, bf16)),
        };
        let different_span_limit = BlockStateShape {
            bank: None,
            history: Some((4096, 1024, 256, 32, bf16)),
        };
        let different_codec = BlockStateShape {
            bank: None,
            history: Some((4096, 1024, 256, 18, f16)),
        };
        assert_eq!(base, base.clone());
        assert_ne!(base, different_page_rows);
        assert_ne!(base, different_span_limit);
        assert_ne!(base, different_codec);
    }
}
