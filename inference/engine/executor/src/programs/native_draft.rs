//! The native separate-draft program (DFlash, DSpark; plan §3.8). One sealed
//! graph per class runs a draft transaction in one device submission:
//!
//! - **Injection.** For every draft layer, the layer's attention over the
//!   entry rows with the target's fused taps (widened to F32) as its hidden
//!   rows and the draft's fusion norm as its input norm, each row attending
//!   only itself: the entry K/V norm, rotary and append every attention
//!   layer uses write the context K/V at the rows' destinations. The
//!   attention's own output is not read.
//! - **Block.** When drafting, one pass over each slot's block
//!   `[anchor, mask, …]`: the raw token embedding, every draft layer's
//!   attention (over the domain's accepted and injected rows, and the block
//!   through the reading row or wholly, per the layer's block attention) and
//!   dense feed-forward, then the output norm and the target's
//!   vocabulary projection of the proposing rows and position-keyed
//!   selection. DSpark then chains its slots: slot `k`'s logits gain the
//!   Markov projection of the token before it, its selection feeds slot
//!   `k + 1`, and a confidence below the threshold declines the proposal.
//!
//! Selections are step-major: proposal `k` of slot `s` is result row
//! `k · slots + s`.

use super::{DeviceSubmission, HeadProgram};
use crate::operators::{self, Mixer};
use crate::{
    completion::CompletionWaiter,
    native::{AttestedDraft, AttestedTarget},
    operators::attention::graph::{
        self as attention_graph, attention_weights, AttentionBlock, AttentionControlPorts,
        AttentionGraphEntries, AttentionHistoryEntries, AttentionWeights, CheckedAttentionEntries,
        ProjectedRows,
    },
    operators::dense_ffn::graph::{self as dense_graph, CheckedDenseEntries, DenseGraphEntries},
    programs::{
        graph::readout::{self, readout_softcap, shapes, write_selection_rows, SelectionPorts},
        graph::{draft::GraphDraft, GraphError},
        native_constants::{
            distinct_storage_bytes, CheckedGraphFamilyResources, CheckedGraphResources,
            ConstantTensors, GraphConstant,
        },
        native_target_graph::{resident_scale, weight, WeightPort},
    },
    DeviceError, DraftProgramPlan, GraphOutputTensor, HeadLaunchCore, InvariantError,
    ModelLoadPlan, NativeGraphOutputLease, NativeGraphWorkspaceLease, ResidentHead, ResidentWeight,
    ResourceLimits, StateStorePlan, SubmitError, ValidatedHeadLaunch,
};
use magnitude_batching::{row_class, TargetBatchUpload};
use magnitude_family_contracts::{
    ActivationDType, BlockAttention, DraftDefinition, DraftEmbedding, DraftMethod, HistoryDomain,
    KeyValue, ModelDefinition, SublayerIndex, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    dense_output, draft_confidence, draft_convolve_input, draft_convolve_residual,
    draft_gated_rows, draft_path_step, draft_top_k, embedding_rows, project_rows,
    readout_features_rows, readout_head_rows, sample_rows, shape_rows, widen_rows,
};
use magnitude_state::LayerRef;
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphFamily,
    NativeGraphMetadata, NativeGraphPlan, NativePort, Tensor, WorkflowTensor,
};
use std::{collections::BTreeMap, rc::Rc};

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native draft program",
        detail: detail.into(),
    })
}
fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}
fn i32_bytes(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn activation(dtype: ActivationDType) -> Element {
    match dtype {
        ActivationDType::F16 => Element::f16(),
        ActivationDType::BF16 => Element::bf16(),
    }
}

/// DSpark's confidence threshold: a slot below it declines its proposal. 0
/// (the default of the reference service) never declines.
const CONFIDENCE_THRESHOLD: f32 = 0.0;

/// One draft graph: the entry rows' class, and when drafting the class of
/// the drafting slots and whether selection is shaped first. Every drafting
/// class drafts the load's proposals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DraftGraphClass {
    pub entry_rows: u64,
    /// 0 for an injection-only transaction.
    pub slots: u64,
    pub shaped: bool,
    /// Whether windowed history layers inject the entry rows. Always for a
    /// drafting class; an injection-only class may skip them when every
    /// entry row falls before the first draft's windows
    /// (`HeadLaunchCore::windowed`).
    pub windowed: bool,
    /// Whether an injection-only entry reads its conditioning from a target
    /// feature output on the device (a prompt chunk's entry drafted behind
    /// the chunk) rather than host-written rows.
    pub device: bool,
}

/// Whether some of the draft's layers keep a windowed history and some do
/// not: only then can an injection skip the windowed layers (every layer
/// windowed leaves nothing to inject).
pub(crate) fn skips_windowed_layers(draft: &DraftDefinition) -> Result<bool, String> {
    let mut windowed = false;
    let mut full = false;
    for index in 0..draft.blocks.len() {
        let paired =
            operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
        let Mixer::Attention(attention) = paired.mixer else {
            return Err("draft layers attend".into());
        };
        match attention.key_value {
            KeyValue::Owned {
                domain: HistoryDomain::Window { .. },
                ..
            } => windowed = true,
            _ => full = true,
        }
    }
    Ok(windowed && full)
}

/// The admitted draft classes: every entry row class injecting only, and
/// every slot class whose blocks (`block` rows each) fit a launch drafting
/// after entry rows up to its proposal rows.
pub(crate) fn draft_graph_classes(
    limits: ResourceLimits,
    proposals: usize,
    draft: &DraftDefinition,
) -> Result<Vec<DraftGraphClass>, String> {
    let block = draft.block_size;
    let skips = skips_windowed_layers(draft)?;
    let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
        .into_iter()
        .map(|rows| rows as u64)
        .collect::<Vec<_>>();
    let max_rows = *row_classes.last().ok_or_else(|| {
        format!(
            "launch row bound {} has no row class",
            limits.max_launch_rows
        )
    })?;
    let slot_classes =
        magnitude_batching::row_classes(limits.max_launch_slots.min(limits.max_launch_rows))
            .into_iter()
            .map(|slots| slots as u64)
            .filter(|slots| slots * block <= max_rows)
            .collect::<Vec<_>>();
    let mut classes = row_classes
        .iter()
        .flat_map(|&entry_rows| {
            [(true, false), (false, false), (true, true), (false, true)]
                .into_iter()
                .filter(move |(windowed, _)| *windowed || skips)
                .map(move |(windowed, device)| DraftGraphClass {
                    entry_rows,
                    slots: 0,
                    shaped: false,
                    windowed,
                    device,
                })
        })
        .collect::<Vec<_>>();
    for &slots in &slot_classes {
        let entry_bound = row_class(
            (slots as usize)
                .saturating_mul(proposals + 1)
                .min(max_rows as usize),
        )
        .ok_or("draft entry row bound has no class")? as u64;
        for &entry_rows in row_classes
            .iter()
            .filter(|rows| **rows >= slots && **rows <= entry_bound)
        {
            for shaped in [false, true] {
                classes.push(DraftGraphClass {
                    entry_rows,
                    slots,
                    shaped,
                    windowed: true,
                    device: false,
                });
            }
        }
    }
    Ok(classes)
}

/// Every class's fixed geometry: the draft, each layer's history rows and
/// slab rows, the block pass's visible spans, and the proposals drafted.
pub(crate) struct DraftGeometry<'d> {
    pub definition: &'d ModelDefinition,
    pub draft: &'d DraftDefinition,
    /// Per draft layer, its history domain's rows and slab rows.
    pub layers: Vec<(u64, u32)>,
    pub segments: u64,
    pub proposals: u64,
}

impl<'d> DraftGeometry<'d> {
    pub(crate) fn new(
        definition: &'d ModelDefinition,
        state: &StateStorePlan,
        proposals: usize,
    ) -> Result<Self, String> {
        let draft = definition
            .draft
            .as_ref()
            .ok_or("draft graphs require a draft")?;
        let layers = (0..draft.blocks.len())
            .map(|index| {
                let layer = LayerRef::Head(u32::try_from(index).map_err(|_| "draft layer index")?);
                let history = state
                    .layer_history(layer)
                    .ok_or("a draft layer has no history in the draft store")?;
                Ok((
                    u64::try_from(history.store.rows).map_err(|_| "draft history rows")?,
                    history.store.slab_rows,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let segments = u64::try_from(
            state
                .span_limit()
                .checked_next_power_of_two()
                .ok_or("draft segment class overflows")?,
        )
        .map_err(|_| "draft segment class exceeds u64")?;
        Ok(Self {
            definition,
            draft,
            layers,
            segments,
            proposals: proposals as u64,
        })
    }
}

/// The draft's entries in the graph's binding form.
pub(crate) struct DraftGraphEntries<'a, G: GraphDraft + 'a> {
    pub layers: Vec<DraftLayerEntries<'a, G>>,
    pub embedding: G::Binding<'a, embedding_rows::Entry>,
    pub head: G::Binding<'a, readout_head_rows::Entry>,
    pub shape: G::Binding<'a, shape_rows::Entry>,
    pub sample: G::Binding<'a, sample_rows::Entry>,
    /// Widens a device-conditioned entry's target features to F32.
    pub widen: G::Binding<'a, widen_rows::Entry>,
    pub markov: Option<MarkovEntries<'a, G>>,
    pub dflash2: Option<Dflash2Entries<'a, G>>,
}

/// DFlash2's unfused layer entries and its candidate path.
pub(crate) struct Dflash2Entries<'a, G: GraphDraft + 'a> {
    pub layers: Vec<Dflash2LayerEntries<'a, G>>,
    pub features: G::Binding<'a, readout_features_rows::Entry>,
    pub hidden: G::Binding<'a, project_rows::Entry>,
    pub convolve_input: G::Binding<'a, draft_convolve_input::Entry>,
    pub convolve_residual: G::Binding<'a, draft_convolve_residual::Entry>,
    pub gated: G::Binding<'a, draft_gated_rows::Entry>,
    pub top_k: G::Binding<'a, draft_top_k::Entry>,
    pub predecessor: G::Binding<'a, embedding_rows::Entry>,
    pub successor: G::Binding<'a, embedding_rows::Entry>,
    pub path: G::Binding<'a, draft_path_step::Entry>,
}

pub(crate) struct Dflash2LayerEntries<'a, G: GraphDraft + 'a> {
    pub attention_norm: G::Binding<'a, readout_features_rows::Entry>,
    pub attention_coefficients: G::Binding<'a, project_rows::Entry>,
    pub query: G::Binding<'a, project_rows::Entry>,
    pub key: G::Binding<'a, project_rows::Entry>,
    pub value: G::Binding<'a, project_rows::Entry>,
    pub output: G::Binding<'a, project_rows::Entry>,
    pub feed_forward_norm: G::Binding<'a, readout_features_rows::Entry>,
    pub feed_forward_coefficients: G::Binding<'a, project_rows::Entry>,
    pub gate: G::Binding<'a, project_rows::Entry>,
    pub up: G::Binding<'a, project_rows::Entry>,
    pub down: G::Binding<'a, project_rows::Entry>,
}

pub(crate) struct DraftLayerEntries<'a, G: GraphDraft + 'a> {
    pub attention: AttentionGraphEntries<'a, G>,
    pub injection: AttentionGraphEntries<'a, G>,
    pub dense: DenseGraphEntries<'a, G>,
}

pub(crate) struct MarkovEntries<'a, G: GraphDraft + 'a> {
    pub embedding: G::Binding<'a, embedding_rows::Entry>,
    pub projection: G::Binding<'a, dense_output::Entry>,
    pub features: G::Binding<'a, readout_features_rows::Entry>,
    pub confidence: G::Binding<'a, draft_confidence::Entry>,
}

impl<'a> DraftGraphEntries<'a, NativeGraph> {
    fn prepared(draft: &'a AttestedDraft, target: &'a AttestedTarget) -> Self {
        Self {
            layers: draft
                .blocks
                .iter()
                .map(|block| DraftLayerEntries {
                    attention: (&block.attention).into(),
                    injection: (&block.injection).into(),
                    dense: (&block.dense).into(),
                })
                .collect(),
            embedding: &draft.embedding,
            head: &draft.head,
            shape: &draft.shape,
            sample: &draft.sample,
            widen: &draft.widen,
            markov: draft.markov.as_ref().map(|markov| MarkovEntries {
                embedding: &markov.embedding,
                projection: &markov.projection,
                features: &markov.features,
                confidence: &markov.confidence,
            }),
            dflash2: draft.dflash2.as_ref().map(|dflash2| Dflash2Entries {
                layers: dflash2
                    .layers
                    .iter()
                    .map(|layer| Dflash2LayerEntries {
                        attention_norm: &layer.attention_norm,
                        attention_coefficients: &layer.attention_coefficients,
                        query: &layer.query,
                        key: &layer.key,
                        value: &layer.value,
                        output: &layer.output,
                        feed_forward_norm: &layer.feed_forward_norm,
                        feed_forward_coefficients: &layer.feed_forward_coefficients,
                        gate: &layer.gate,
                        up: &layer.up,
                        down: &layer.down,
                    })
                    .collect(),
                features: &dflash2.features,
                hidden: &dflash2.hidden,
                convolve_input: &dflash2.convolve_input,
                convolve_residual: &dflash2.convolve_residual,
                gated: &dflash2.gated,
                top_k: &dflash2.top_k,
                predecessor: &dflash2.predecessor,
                successor: &dflash2.successor,
                path: &dflash2.path,
            }),
        }
    }
}

/// The draft's entries as element assignments of its program plan, for the
/// metadata-only (checked) graphs.
struct CheckedDraftEntries {
    layers: Vec<(
        CheckedAttentionEntries,
        CheckedAttentionEntries,
        CheckedDenseEntries,
    )>,
    embedding: [(&'static str, Element); 2],
    head: [(&'static str, Element); 3],
    selection: [(&'static str, Element); 0],
    widen: [(&'static str, Element); 1],
    markov: Option<CheckedMarkovEntries>,
    dflash2: Option<CheckedDflash2Entries>,
}

type Assignments<const N: usize> = [(&'static str, Element); N];

struct CheckedDflash2Entries {
    layers: Vec<CheckedDflash2Layer>,
    features: Assignments<2>,
    hidden: Assignments<3>,
    activation: Assignments<1>,
    none: Assignments<0>,
    predecessor: Assignments<2>,
    successor: Assignments<2>,
}

struct CheckedDflash2Layer {
    attention_norm: Assignments<2>,
    attention_coefficients: Assignments<3>,
    query: Assignments<3>,
    key: Assignments<3>,
    value: Assignments<3>,
    output: Assignments<3>,
    feed_forward_norm: Assignments<2>,
    feed_forward_coefficients: Assignments<3>,
    gate: Assignments<3>,
    up: Assignments<3>,
    down: Assignments<3>,
}

impl CheckedDflash2Entries {
    fn new(plan: &DraftProgramPlan, binding: &crate::Dflash2Binding) -> Self {
        let activation = plan.activation();
        let projection =
            |weight: Element, output: Element| [("A", activation), ("W", weight), ("Y", output)];
        let (a, f32) = (activation, Element::f32());
        Self {
            layers: plan
                .blocks()
                .iter()
                .zip(&binding.convolutions)
                .map(|(block, [attention_coefficients, dense_coefficients])| {
                    let (attention, dense) = (block.attention, block.feed_forward);
                    CheckedDflash2Layer {
                        attention_norm: [("NW", attention.norm), ("A", a)],
                        attention_coefficients: projection(*attention_coefficients, f32),
                        query: projection(attention.query, a),
                        key: projection(attention.key, a),
                        value: projection(attention.value, a),
                        output: projection(attention.output, f32),
                        feed_forward_norm: [("NW", dense.norm), ("A", a)],
                        feed_forward_coefficients: projection(*dense_coefficients, f32),
                        gate: projection(dense.gate, a),
                        up: projection(dense.up, a),
                        down: projection(dense.down, f32),
                    }
                })
                .collect(),
            features: [("NW", plan.output_norm()), ("A", a)],
            hidden: projection(binding.selector.hidden, a),
            activation: [("A", a)],
            none: [],
            predecessor: [("EW", binding.selector.predecessor), ("A", a)],
            successor: [("EW", binding.selector.successor), ("A", a)],
        }
    }

    fn entries(&self) -> Dflash2Entries<'_, NativeGraphMetadata> {
        Dflash2Entries {
            layers: self
                .layers
                .iter()
                .map(|layer| Dflash2LayerEntries {
                    attention_norm: &layer.attention_norm[..],
                    attention_coefficients: &layer.attention_coefficients[..],
                    query: &layer.query[..],
                    key: &layer.key[..],
                    value: &layer.value[..],
                    output: &layer.output[..],
                    feed_forward_norm: &layer.feed_forward_norm[..],
                    feed_forward_coefficients: &layer.feed_forward_coefficients[..],
                    gate: &layer.gate[..],
                    up: &layer.up[..],
                    down: &layer.down[..],
                })
                .collect(),
            features: &self.features[..],
            hidden: &self.hidden[..],
            convolve_input: &self.activation[..],
            convolve_residual: &self.none[..],
            gated: &self.activation[..],
            top_k: &self.none[..],
            predecessor: &self.predecessor[..],
            successor: &self.successor[..],
            path: &self.activation[..],
        }
    }
}

struct CheckedMarkovEntries {
    embedding: [(&'static str, Element); 2],
    projection: [(&'static str, Element); 2],
    features: [(&'static str, Element); 2],
    confidence: [(&'static str, Element); 1],
}

impl CheckedDraftEntries {
    fn new(plan: &DraftProgramPlan) -> Self {
        let activation = plan.activation();
        Self {
            layers: plan
                .blocks()
                .iter()
                .map(|block| {
                    (
                        CheckedAttentionEntries::new(block.attention),
                        CheckedAttentionEntries::new(block.injection),
                        CheckedDenseEntries::new(block.feed_forward),
                    )
                })
                .collect(),
            embedding: [("EW", plan.embedding().table), ("A", activation)],
            head: [
                ("NW", plan.output_norm()),
                ("OW", plan.projection()),
                ("A", activation),
            ],
            selection: [],
            widen: [("A", activation)],
            markov: plan.markov().map(|markov| CheckedMarkovEntries {
                embedding: [("EW", markov.embedding), ("A", activation)],
                projection: [("DW", markov.projection), ("A", activation)],
                features: [("NW", plan.output_norm()), ("A", activation)],
                confidence: [("A", activation)],
            }),
            dflash2: plan
                .dflash2()
                .map(|binding| CheckedDflash2Entries::new(plan, binding)),
        }
    }

    fn entries(&self) -> Result<DraftGraphEntries<'_, NativeGraphMetadata>, String> {
        Ok(DraftGraphEntries {
            layers: self
                .layers
                .iter()
                .map(|(attention, injection, dense)| {
                    Ok(DraftLayerEntries {
                        attention: attention.entries()?,
                        injection: injection.entries()?,
                        dense: dense.entries(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
            embedding: &self.embedding[..],
            head: &self.head[..],
            shape: &self.selection[..],
            sample: &self.selection[..],
            widen: &self.widen[..],
            markov: self.markov.as_ref().map(|markov| MarkovEntries {
                embedding: &markov.embedding[..],
                projection: &markov.projection[..],
                features: &markov.features[..],
                confidence: &markov.confidence[..],
            }),
            dflash2: self.dflash2.as_ref().map(CheckedDflash2Entries::entries),
        })
    }
}

/// The storage and bound constants of the draft's graph family over
/// `classes`, from metadata alone: each class is sealed exactly, and the
/// family holds the largest of each arena.
pub(crate) fn checked_draft_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    plan: &DraftProgramPlan,
    geometry: &DraftGeometry<'_>,
    classes: impl IntoIterator<Item = DraftGraphClass>,
) -> Result<CheckedGraphResources, GraphError> {
    let checked = CheckedDraftEntries::new(plan);
    let mut family = CheckedGraphFamilyResources::new();
    for class in classes {
        let context = || format!("draft graph class {class:?}");
        let parts = draft_graph(
            NativeGraphMetadata::new(backend),
            checked.entries()?,
            load,
            geometry,
            class,
        )
        .map_err(|error| error.context(context()))?;
        let storage = GraphDraft::seal(parts.plan).map_err(|error| error.context(context()))?;
        family.include(storage, parts.constants);
    }
    Ok(family.finish()?)
}

/// One layer's per-run ports in one pass: its attention controls and its
/// history planes, in plane-descriptor order.
struct LayerPorts {
    controls: AttentionControlPorts,
    planes: Vec<NativePort>,
}

struct BlockPorts {
    /// `[slots · block, 2]` (token, status) rows.
    tokens: NativePort,
    /// The block row of each proposal, step-major.
    head_rows: NativePort,
    layers: Vec<LayerPorts>,
    /// One selection for every proposal (DFlash), or one per step (DSpark).
    selections: Vec<SelectionPorts>,
    /// DSpark: each slot's anchor, the token its first step conditions on.
    anchors: Option<NativePort>,
    /// Weights whose rows are vocabulary entries of the readout (the target's
    /// projection, DSpark's Markov projection), bound to their leading
    /// `readout_vocabulary` rows.
    leading: Vec<(WeightRole, NativePort)>,
    readout_vocabulary: u64,
}

/// A port over the leading `rows` rows of `role`'s planned weight, whose
/// rows are `width` wide.
fn leading_port<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    role: WeightRole,
    rows: u64,
    width: u64,
    leading: &mut Vec<(WeightRole, NativePort)>,
) -> Result<WorkflowTensor, GraphError> {
    let plan = load
        .weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| format!("planned weight {role:?} is absent"))?;
    let port = graph.port(plan.resident, &[rows, width])?;
    let tensor = port.tensor().clone();
    leading.push((role, port));
    Ok(tensor)
}

/// Where a draft graph's `[entry rows, hidden]` conditioning comes from.
#[derive(Clone)]
enum DraftConditioning {
    /// F32 rows the host writes.
    Rows(NativePort),
    /// A bound target feature output (activation rows), widened in the
    /// graph.
    Features(NativePort),
}

struct DraftGraphParts<P> {
    plan: P,
    conditioning: DraftConditioning,
    /// Per draft layer, its injection pass; `None` for a windowed layer an
    /// injection-only class without windowed layers skips.
    injection: Vec<Option<LayerPorts>>,
    block: Option<BlockPorts>,
    constants: Vec<GraphConstant>,
    weights: Vec<(WeightPort, NativePort)>,
    /// Selections `[proposals · slots, 2]` when drafting; otherwise the
    /// last injection's rows, exported so the graph has a result.
    output: WorkflowTensor,
}

/// The graph's absent scale, declared on its first read (`slot`).
fn absent<G: GraphDraft>(
    graph: &mut G,
    constants: &mut Vec<GraphConstant>,
    slot: &mut Option<WorkflowTensor>,
) -> Result<WorkflowTensor, GraphError> {
    if let Some(absent) = slot {
        return Ok(absent.clone());
    }
    let absent = GraphConstant::absent_scale(graph, constants)?;
    *slot = Some(absent.clone());
    Ok(absent)
}

fn draft_sublayer(block: usize, sublayer: u32) -> Result<WeightScope, String> {
    Ok(WeightScope::DraftSublayer(SublayerIndex {
        block: u32::try_from(block).map_err(|_| "draft block index exceeds u32")?,
        sublayer,
    }))
}

fn draft_graph<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: DraftGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    geometry: &DraftGeometry<'_>,
    class: DraftGraphClass,
) -> Result<DraftGraphParts<G>, GraphError> {
    let decoder = &geometry.definition.decoder;
    let draft = geometry.draft;
    let (hidden, vocabulary) = (decoder.hidden, decoder.vocabulary);
    let activation = activation(decoder.activation_dtype);
    let epsilon = draft.output_norm.epsilon as f32;
    if entries.layers.len() != draft.blocks.len() || geometry.layers.len() != draft.blocks.len() {
        return Err("draft entries disagree with the draft's layers".into());
    }
    let mut weights = Vec::new();
    let mut constants = Vec::new();
    let fusion_norm = weight(
        &mut graph,
        load,
        WeightScope::Draft,
        WeightKind::DraftFusionNorm,
        &mut weights,
    )?;
    // Each layer's attention weights for injection, whose input norm is the
    // fusion norm; the block pass reads the same weights with the layer's
    // own input norm.
    let mut layer_weights = Vec::with_capacity(draft.blocks.len());
    for index in 0..draft.blocks.len() {
        let paired =
            operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
        let Mixer::Attention(attention) = paired.mixer else {
            return Err("draft layers attend".into());
        };
        let windowed_layer = matches!(
            attention.key_value,
            KeyValue::Owned {
                domain: HistoryDomain::Window { .. },
                ..
            }
        );
        if windowed_layer && !class.windowed {
            layer_weights.push(None);
            continue;
        }
        let shape =
            operators::attention::shape(hidden, attention).map_err(|error| error.to_string())?;
        let scope = draft_sublayer(index, 0)?;
        let injection =
            attention_weights(
                &shape,
                attention,
                class.slots > 0,
                false,
                |kind| match kind {
                    WeightKind::InputNorm => Ok(fusion_norm.clone()),
                    kind => weight(&mut graph, load, scope, kind, &mut weights),
                },
            )?;
        layer_weights.push(Some((attention, shape, injection)));
    }
    fn attention_block<'o>(
        operator: &'o magnitude_family_contracts::Attention,
        shape: crate::AttentionShape,
        rows: u64,
        segments: u64,
        (history_rows, slab_rows): (u64, u32),
        epsilon: f32,
        activation: Element,
        inject_only: bool,
    ) -> AttentionBlock<'o> {
        AttentionBlock {
            rows,
            segments,
            history_rows,
            slab_rows,
            shape,
            operator,
            epsilon,
            head_epsilon: epsilon,
            post_norm_epsilon: 0.0,
            post_norm_scale: 1.0,
            activation,
            inject_only,
        }
    }

    // Injection: each entry row appends its context K/V at its destination.
    // The host writes the rows (an input of the first injection's
    // projection; every layer's projection reads the same rows), or a
    // device-conditioned class widens its bound target features.
    let (first, (_, first_shape, _)) = layer_weights
        .iter()
        .enumerate()
        .find_map(|(index, layer)| layer.as_ref().map(|layer| (index, layer)))
        .ok_or("a draft has no injected layers")?;
    let (conditioning, conditioning_rows) = if class.device {
        let features =
            graph.port_with_class_extent(activation, &[class.entry_rows, hidden], 0, "M")?;
        let widened = graph
            .enqueue(
                entries.widen,
                &[("M", class.entry_rows), ("D", hidden)],
                widen_rows::WorkflowArgs {
                    rows: features.tensor().into(),
                },
            )?
            .value;
        (DraftConditioning::Features(features), widened)
    } else {
        let rows = graph.input_for(
            entries.layers[first].injection.project,
            "hidden",
            &first_shape.project_dimensions(class.entry_rows),
        )?;
        let tensor = rows.tensor().clone();
        (DraftConditioning::Rows(rows), tensor)
    };
    let mut injection = Vec::with_capacity(draft.blocks.len());
    let mut injected = None;
    for (index, (layer_weight, layer)) in layer_weights.iter().zip(&entries.layers).enumerate() {
        let Some((operator, shape, weights_of)) = layer_weight else {
            injection.push(None);
            continue;
        };
        let (mixed, state, controls) = attention_graph::attention(
            &mut graph,
            layer.injection,
            weights_of,
            &mut constants,
            &conditioning_rows,
            attention_block(
                operator,
                *shape,
                class.entry_rows,
                1,
                geometry.layers[index],
                epsilon,
                activation,
                true,
            ),
        )?;
        injection.push(Some(LayerPorts {
            controls,
            planes: state.planes,
        }));
        injected = Some(mixed);
    }
    let injected = injected.ok_or("a draft has no layers")?;
    if class.slots == 0 {
        graph.export(&injected)?;
        return Ok(DraftGraphParts {
            plan: graph,
            conditioning,
            injection,
            block: None,
            constants,
            weights,
            output: injected,
        });
    }

    // The block pass. Its unscaled weights bind the absent scale, declared
    // by its first reader: an injection-only graph, or one whose every
    // reader binds a resident scale, reads none, and a sealed graph binds no
    // unread port.
    let mut absent_scale = None;
    let slots = class.slots;
    let proposals = geometry.proposals;
    let block_rows = draft.block_rows(proposals);
    let rows = slots * block_rows;
    let outputs = proposals * slots;
    let table = match &draft.embedding {
        DraftEmbedding::Target => weight(
            &mut graph,
            load,
            WeightScope::Target,
            WeightKind::Embedding,
            &mut weights,
        )?,
        DraftEmbedding::Own(_) => weight(
            &mut graph,
            load,
            WeightScope::Draft,
            WeightKind::Embedding,
            &mut weights,
        )?,
    };
    let embedding_dims = [("M", rows), ("V", vocabulary), ("D", hidden)];
    let tokens = graph.input_for(entries.embedding, "tokens", &embedding_dims)?;
    let mut residual = graph
        .enqueue(
            entries.embedding,
            &embedding_dims,
            embedding_rows::WorkflowArgs {
                table: (&table).into(),
                tokens: tokens.tensor().into(),
                scale: 1.0,
                normalize: 0,
                epsilon,
            },
        )?
        .r1;
    let mut layers = Vec::with_capacity(draft.blocks.len());
    // DFlash2's unfused layers read their normed rows through the identity
    // row map of the block rows.
    let dflash2 = match (&draft.method, &entries.dflash2) {
        (DraftMethod::DFlash2 { kernel, group, .. }, Some(dflash2)) => {
            let identity = GraphConstant::identity_for_class(&mut graph, rows, Some("M"))?;
            let rows_map = identity.port().tensor().clone();
            constants.push(identity);
            Some((dflash2, *kernel, *group, rows_map))
        }
        (DraftMethod::DFlash2 { .. }, None) | (_, Some(_)) => {
            return Err("draft method and entries disagree".into())
        }
        (DraftMethod::DFlash | DraftMethod::DSpark { .. }, None) => None,
    };
    for (index, (layer_weight, layer)) in layer_weights.iter().zip(&entries.layers).enumerate() {
        // A drafting class injects every layer.
        let (operator, shape, injection_weights) = layer_weight
            .as_ref()
            .ok_or("a drafting class skips a layer's injection")?;
        if let Some((dflash2, kernel, group, rows_map)) = &dflash2 {
            let absent_scale = absent(&mut graph, &mut constants, &mut absent_scale)?;
            let ports;
            (residual, ports) = dflash2_layer(
                &mut graph,
                Dflash2Layer {
                    entries: &dflash2.layers[index],
                    convolve_input: dflash2.convolve_input,
                    convolve_residual: dflash2.convolve_residual,
                    gated: dflash2.gated,
                    mix: layer.attention.history,
                    index,
                    operator,
                    shape: *shape,
                    attention_weights: injection_weights,
                    rows,
                    rows_map,
                    block_rows,
                    kernel: *kernel,
                    group: *group,
                    hidden,
                    intermediate: dense_intermediate(draft, index)?,
                    epsilon,
                    activation,
                    segments: geometry.segments,
                    history: geometry.layers[index],
                },
                load,
                &mut weights,
                &mut constants,
                &absent_scale,
                &residual,
            )?;
            layers.push(ports);
            continue;
        }
        let input_norm = weight(
            &mut graph,
            load,
            draft_sublayer(index, 0)?,
            WeightKind::InputNorm,
            &mut weights,
        )?;
        let block_weights = AttentionWeights {
            input_norm,
            query: injection_weights.query.clone(),
            gate: injection_weights.gate.clone(),
            key: injection_weights.key.clone(),
            value: injection_weights.value.clone(),
            query_norm: injection_weights.query_norm.clone(),
            key_norm: injection_weights.key_norm.clone(),
            output: injection_weights.output.clone(),
            post_norm: None,
        };
        let (mixed, state, controls) = attention_graph::attention(
            &mut graph,
            layer.attention,
            &block_weights,
            &mut constants,
            &residual,
            attention_block(
                operator,
                *shape,
                rows,
                geometry.segments,
                geometry.layers[index],
                epsilon,
                activation,
                false,
            ),
        )?;
        layers.push(LayerPorts {
            controls,
            planes: state.planes,
        });
        let paired =
            operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
        let Some(operators::FeedForward::Dense(dense)) =
            paired.feed_forward.map(|sublayer| sublayer.op)
        else {
            return Err("draft layers have a dense feed-forward".into());
        };
        residual = dense_graph::dense(
            &mut graph,
            DenseGraphEntries {
                expand: layer.dense.expand,
                output: layer.dense.output,
            },
            load,
            draft_sublayer(index, 1)?,
            &mut weights,
            &mut constants,
            &mixed,
            rows,
            epsilon,
            operators::dense_ffn::activation_code(dense.up.activation()),
            0.0,
            1.0,
        )?;
    }
    let output_norm = weight(
        &mut graph,
        load,
        WeightScope::Draft,
        WeightKind::OutputNorm,
        &mut weights,
    )?;
    let readout_vocabulary = crate::native::draft_readout_vocabulary(
        matches!(draft.method, DraftMethod::DSpark { .. }),
        vocabulary,
    );
    let mut leading = Vec::new();
    let projection = leading_port(
        &mut graph,
        load,
        WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::Output,
        },
        readout_vocabulary,
        hidden,
        &mut leading,
    )?;
    // The projection's accumulator-scale port: its resident second-level
    // scale, else the absent scale.
    let projection_scale = resident_scale(
        &mut graph,
        load,
        WeightScope::Target,
        WeightKind::Output,
        &mut weights,
    )?;
    let (projection_scale, scale_extent) = match projection_scale {
        Some(scale) => (scale, 1),
        None => (absent(&mut graph, &mut constants, &mut absent_scale)?, 0),
    };
    let head_dims = [
        ("M", rows),
        ("O", outputs),
        ("V", readout_vocabulary),
        ("D", hidden),
        ("WS", scale_extent),
    ];
    let head_rows = graph.input_for(entries.head, "out_rows", &head_dims)?;
    let logits = graph
        .enqueue(
            entries.head,
            &head_dims,
            readout_head_rows::WorkflowArgs {
                hidden: (&residual).into(),
                norm: (&output_norm).into(),
                weight: (&projection).into(),
                out_rows: head_rows.tensor().into(),
                epsilon,
                softcap: readout_softcap(decoder),
                weight_scale: (&projection_scale).into(),
            },
        )?
        .value;
    let result = graph.local_for(
        entries.sample,
        "result",
        &[("M", outputs), ("V", readout_vocabulary)],
    )?;
    let mut selections = Vec::new();
    let anchors = match (&draft.method, &entries.markov) {
        (DraftMethod::DFlash, None) => {
            let mut all = result.tensor().slice_leading(0, outputs);
            selections.push(readout::sample(
                &mut graph,
                entries.shape,
                entries.sample,
                readout_vocabulary,
                (&logits).into(),
                outputs,
                class.shaped,
                (&mut all).into(),
            )?);
            None
        }
        (DraftMethod::DSpark { markov, .. }, Some(chain)) => {
            let rank = markov.rank;
            let markov_table = weight(
                &mut graph,
                load,
                WeightScope::Draft,
                WeightKind::MarkovEmbedding,
                &mut weights,
            )?;
            // Its bias adds one logit per readout vocabulary row.
            let markov_projection = leading_port(
                &mut graph,
                load,
                WeightRole {
                    scope: WeightScope::Draft,
                    kind: WeightKind::MarkovProjection,
                },
                readout_vocabulary,
                rank,
                &mut leading,
            )?;
            let confidence_weight = weight(
                &mut graph,
                load,
                WeightScope::Draft,
                WeightKind::ConfidenceWeight,
                &mut weights,
            )?;
            let confidence_bias = weight(
                &mut graph,
                load,
                WeightScope::Draft,
                WeightKind::ConfidenceBias,
                &mut weights,
            )?;
            let memory_dims = [("M", slots), ("V", vocabulary), ("D", rank)];
            let anchors = graph.input_for(chain.embedding, "tokens", &memory_dims)?;
            for step in 0..proposals {
                let rows_of_step = GraphConstant::i32(
                    &mut graph,
                    &(step * slots..(step + 1) * slots)
                        .map(|row| i32::try_from(row).map_err(|_| "draft output row exceeds i32"))
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                let previous = match step {
                    0 => anchors.tensor().slice_leading(0, slots),
                    _ => result
                        .tensor()
                        .slice_leading((step - 1) * slots, step * slots),
                };
                let memory = graph
                    .enqueue(
                        chain.embedding,
                        &memory_dims,
                        embedding_rows::WorkflowArgs {
                            table: (&markov_table).into(),
                            tokens: (&previous).into(),
                            scale: 1.0,
                            normalize: 0,
                            epsilon,
                        },
                    )?
                    .r0;
                let absent_scale = absent(&mut graph, &mut constants, &mut absent_scale)?;
                let biased = graph
                    .enqueue(
                        chain.projection,
                        &[
                            ("M", outputs),
                            ("O", slots),
                            ("H", readout_vocabulary),
                            ("F", rank),
                            ("DS", 0),
                        ],
                        dense_output::WorkflowArgs {
                            residual: (&logits).into(),
                            product: (&memory).into(),
                            down_weight: (&markov_projection).into(),
                            out_rows: rows_of_step.port().tensor().into(),
                            down_scale: (&absent_scale).into(),
                        },
                    )?
                    .value;
                let mut step_result = result
                    .tensor()
                    .slice_leading(step * slots, (step + 1) * slots);
                selections.push(readout::sample(
                    &mut graph,
                    entries.shape,
                    entries.sample,
                    readout_vocabulary,
                    (&biased).into(),
                    slots,
                    class.shaped,
                    (&mut step_result).into(),
                )?);
                let step_rows = head_rows
                    .tensor()
                    .slice_leading(step * slots, (step + 1) * slots);
                let features = graph
                    .enqueue(
                        chain.features,
                        &[("M", rows), ("O", slots), ("D", hidden)],
                        readout_features_rows::WorkflowArgs {
                            hidden: (&residual).into(),
                            norm: (&output_norm).into(),
                            out_rows: (&step_rows).into(),
                            epsilon,
                        },
                    )?
                    .value;
                graph.enqueue(
                    chain.confidence,
                    &[("S", slots), ("D", hidden), ("R", rank)],
                    draft_confidence::WorkflowArgs {
                        features: (&features).into(),
                        memory: (&memory).into(),
                        weight: (&confidence_weight).into(),
                        bias: (&confidence_bias).into(),
                        threshold: CONFIDENCE_THRESHOLD,
                        selection: (&mut step_result).into(),
                    },
                )?;
                constants.push(rows_of_step);
            }
            Some(anchors)
        }
        (DraftMethod::DFlash2 { selector, .. }, None) => {
            let (path, ..) = dflash2
                .as_ref()
                .ok_or("a DFlash2 draft graph has no DFlash2 entries")?;
            let absent_scale = absent(&mut graph, &mut constants, &mut absent_scale)?;
            Some(dflash2_path(
                &mut graph,
                path,
                load,
                &mut weights,
                &absent_scale,
                Dflash2Path {
                    residual: &residual,
                    output_norm: &output_norm,
                    head_rows: head_rows.tensor(),
                    logits: &logits,
                    result: result.tensor(),
                    rows,
                    slots,
                    proposals,
                    vocabulary,
                    readout_vocabulary,
                    hidden,
                    rank: selector.rank,
                    top_k: selector.top_k,
                    epsilon,
                },
            )?)
        }
        _ => return Err("draft method and entries disagree".into()),
    };
    let output = result.tensor().clone();
    graph.export(&output)?;
    Ok(DraftGraphParts {
        plan: graph,
        conditioning,
        injection,
        block: Some(BlockPorts {
            tokens,
            head_rows,
            layers,
            selections,
            anchors,
            leading,
            readout_vocabulary,
        }),
        constants,
        weights,
        output,
    })
}

/// Draft layer `index`'s feed-forward expansion width.
fn dense_intermediate(draft: &DraftDefinition, index: usize) -> Result<u64, String> {
    let paired = operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
    match paired.feed_forward.map(|sublayer| sublayer.op) {
        Some(operators::FeedForward::Dense(dense)) => Ok(dense.intermediate),
        _ => Err("draft layers have a dense feed-forward".into()),
    }
}

/// One DFlash2 draft layer of the block pass and what it reads.
struct Dflash2Layer<'a, 'o, G: GraphDraft + 'a> {
    entries: &'o Dflash2LayerEntries<'a, G>,
    convolve_input: G::Binding<'a, draft_convolve_input::Entry>,
    convolve_residual: G::Binding<'a, draft_convolve_residual::Entry>,
    gated: G::Binding<'a, draft_gated_rows::Entry>,
    /// The layer's attention entry of its history codec.
    mix: AttentionHistoryEntries<'a, G>,
    index: usize,
    operator: &'o magnitude_family_contracts::Attention,
    shape: crate::AttentionShape,
    /// The layer's projection and head-norm weight ports.
    attention_weights: &'o AttentionWeights,
    rows: u64,
    /// The identity row map of the block rows.
    rows_map: &'o WorkflowTensor,
    /// Rows of each slot's block (`DraftDefinition::block_rows`).
    block_rows: u64,
    kernel: u64,
    group: u64,
    hidden: u64,
    /// The feed-forward's expansion width.
    intermediate: u64,
    epsilon: f32,
    activation: Element,
    segments: u64,
    /// The layer's history rows and slab rows.
    history: (u64, u32),
}

/// One DFlash2 layer over the block rows (upstream `Qwen3DFlashDecoderLayer`
/// with its convolutions): for the attention and then the feed-forward, the
/// normed rows project to their convolution coefficients, the half-0
/// convolution feeds the operator's plain projections, and the half-1
/// convolution of the operator's F32 output joins the residual. The
/// attention appends nothing (its destinations are the block pass's −1) and
/// reads the injected context and the whole fresh block.
#[allow(clippy::too_many_arguments)]
fn dflash2_layer<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    layer: Dflash2Layer<'a, '_, G>,
    load: &ModelLoadPlan,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    absent_scale: &WorkflowTensor,
    residual: &WorkflowTensor,
) -> Result<(WorkflowTensor, LayerPorts), GraphError> {
    let Dflash2Layer {
        entries,
        rows,
        hidden,
        epsilon,
        ..
    } = layer;
    let groups = hidden / layer.group;
    let coefficients = 2 * layer.kernel * groups;
    let convolution_dims = [
        ("M", rows),
        ("G", groups),
        ("C", layer.group),
        ("K", layer.kernel),
    ];
    let block = u32::try_from(layer.block_rows).map_err(|_| "draft block exceeds u32")?;
    let project = |graph: &mut G,
                   entry: G::Binding<'a, project_rows::Entry>,
                   source: seismic::WorkflowTensorRef<'_>,
                   weight: &WorkflowTensor,
                   (inputs, outputs): (u64, u64)| {
        graph
            .enqueue(
                entry,
                &[("M", rows), ("K", inputs), ("N", outputs), ("WS", 0)],
                project_rows::WorkflowArgs {
                    source,
                    weight: weight.into(),
                    weight_scale: absent_scale.into(),
                },
            )
            .map(|projected| projected.value)
    };
    // One sublayer's prologue: its normed rows, their coefficients and the
    // half-0 convolution the operator reads.
    let prologue = |graph: &mut G,
                    weights: &mut Vec<(WeightPort, NativePort)>,
                    residual: &WorkflowTensor,
                    sublayer: u32,
                    norm_entry: G::Binding<'a, readout_features_rows::Entry>,
                    coefficient_entry: G::Binding<'a, project_rows::Entry>| {
        let scope = draft_sublayer(layer.index, sublayer)?;
        let norm = weight(graph, load, scope, WeightKind::InputNorm, weights)?;
        let base = weight(graph, load, scope, WeightKind::ConvolutionBase, weights)?;
        let projection = weight(
            graph,
            load,
            scope,
            WeightKind::ConvolutionProjection,
            weights,
        )?;
        let normed = graph
            .enqueue(
                norm_entry,
                &[("M", rows), ("O", rows), ("D", hidden)],
                readout_features_rows::WorkflowArgs {
                    hidden: residual.into(),
                    norm: (&norm).into(),
                    out_rows: layer.rows_map.into(),
                    epsilon,
                },
            )?
            .value;
        let dynamic = project(
            graph,
            coefficient_entry,
            (&normed).into(),
            &projection,
            (hidden, coefficients),
        )?;
        let dynamic = dynamic.reshape(&[rows, 2, layer.kernel, groups]);
        let convolved = graph
            .enqueue(
                layer.convolve_input,
                &convolution_dims,
                draft_convolve_input::WorkflowArgs {
                    input: (&normed).into(),
                    dynamic: (&dynamic).into(),
                    base: (&base).into(),
                    block,
                },
            )?
            .value;
        Ok::<_, GraphError>((convolved, dynamic, base))
    };
    let finish =
        |graph: &mut G,
         residual: &WorkflowTensor,
         output: &WorkflowTensor,
         (dynamic, base): (&seismic::WorkflowTensorView, &WorkflowTensor)| {
            graph
                .enqueue(
                    layer.convolve_residual,
                    &convolution_dims,
                    draft_convolve_residual::WorkflowArgs {
                        residual: residual.into(),
                        output: output.into(),
                        dynamic: dynamic.into(),
                        base: base.into(),
                        block,
                    },
                )
                .map(|finished| finished.value)
        };

    // Attention: plain query, key and value projections of the convolved
    // rows, the history codec's attention, and the output projection.
    let (convolved, dynamic, base) = prologue(
        graph,
        weights,
        residual,
        0,
        entries.attention_norm,
        entries.attention_coefficients,
    )?;
    let shape = layer.shape;
    let (heads, width) = (shape.heads(), shape.width);
    let key_width = shape.kv_heads * width;
    let attention = layer.attention_weights;
    let (Some(key_weight), Some(value_weight)) = (&attention.key, &attention.value) else {
        return Err("a DFlash2 draft layer projects its keys and values".into());
    };
    let query = project(
        graph,
        entries.query,
        (&convolved).into(),
        &attention.query,
        (hidden, heads * width),
    )?;
    let key = project(
        graph,
        entries.key,
        (&convolved).into(),
        key_weight,
        (hidden, key_width),
    )?;
    let value = project(
        graph,
        entries.value,
        (&convolved).into(),
        value_weight,
        (hidden, key_width),
    )?;
    let (attended, state, controls) = attention_graph::mix(
        graph,
        layer.mix,
        (&attention.query_norm, &attention.key_norm),
        constants,
        &ProjectedRows {
            query: query.reshape(&[rows, heads, width]),
            // No gate: zero columns of every row.
            gate: query.slice_leading(0, 0).reshape(&[rows, heads, 0]),
            key: key.reshape(&[1, rows, key_width]),
            value: value.reshape(&[1, rows, key_width]),
        },
        &AttentionBlock {
            rows,
            segments: layer.segments,
            history_rows: layer.history.0,
            slab_rows: layer.history.1,
            shape,
            operator: layer.operator,
            epsilon,
            head_epsilon: epsilon,
            post_norm_epsilon: 0.0,
            post_norm_scale: 1.0,
            activation: layer.activation,
            inject_only: false,
        },
    )?;
    let output = project(
        graph,
        entries.output,
        (&attended.reshape(&[rows, heads * width])).into(),
        attention
            .output
            .as_ref()
            .ok_or("draft block output weight is absent")?,
        (heads * width, hidden),
    )?;
    let residual = finish(graph, residual, &output, (&dynamic, &base))?;

    // Feed-forward: gate and up projections of the convolved rows, their
    // gated product and the down projection.
    let (convolved, dynamic, base) = prologue(
        graph,
        weights,
        &residual,
        1,
        entries.feed_forward_norm,
        entries.feed_forward_coefficients,
    )?;
    let scope = draft_sublayer(layer.index, 1)?;
    let gate_weight = weight(graph, load, scope, WeightKind::DenseGate, weights)?;
    let up_weight = weight(graph, load, scope, WeightKind::DenseUp, weights)?;
    let down_weight = weight(graph, load, scope, WeightKind::DenseDown, weights)?;
    let features = layer.intermediate;
    let gate = project(
        graph,
        entries.gate,
        (&convolved).into(),
        &gate_weight,
        (hidden, features),
    )?;
    let up = project(
        graph,
        entries.up,
        (&convolved).into(),
        &up_weight,
        (hidden, features),
    )?;
    let product = graph
        .enqueue(
            layer.gated,
            &[("M", rows), ("F", features)],
            draft_gated_rows::WorkflowArgs {
                gate: (&gate).into(),
                up: (&up).into(),
            },
        )?
        .value;
    let down = project(
        graph,
        entries.down,
        (&product).into(),
        &down_weight,
        (features, hidden),
    )?;
    let residual = finish(graph, &residual, &down, (&dynamic, &base))?;
    Ok((
        residual,
        LayerPorts {
            controls,
            planes: state.planes,
        },
    ))
}

/// The block pass's rows DFlash2's candidate path reads.
struct Dflash2Path<'t> {
    residual: &'t WorkflowTensor,
    output_norm: &'t WorkflowTensor,
    /// The block row of each proposal, step-major.
    head_rows: &'t WorkflowTensor,
    /// The proposing rows' readout logits `[outputs, readout_vocabulary]`.
    logits: &'t WorkflowTensor,
    /// The selections `[outputs, 2]`, step-major.
    result: &'t WorkflowTensor,
    rows: u64,
    slots: u64,
    proposals: u64,
    /// The codebooks' rows: any token id.
    vocabulary: u64,
    readout_vocabulary: u64,
    hidden: u64,
    rank: u64,
    top_k: u64,
    epsilon: f32,
}

/// DFlash2's ordered candidate path (upstream `CandidateSelector.select`,
/// greedy): every proposing row's top-k candidates and their logits, its
/// output-normed row's selector projection and its candidates' successor
/// codes; then per step, the predecessor code of the token each slot
/// follows (its anchor, then its previous selection) and the step's
/// selection. Returns the anchors' input port.
fn dflash2_path<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: &Dflash2Entries<'a, G>,
    load: &ModelLoadPlan,
    weights: &mut Vec<(WeightPort, NativePort)>,
    absent_scale: &WorkflowTensor,
    path: Dflash2Path<'_>,
) -> Result<NativePort, GraphError> {
    let Dflash2Path {
        slots,
        rank,
        top_k,
        vocabulary,
        readout_vocabulary,
        hidden,
        ..
    } = path;
    let outputs = path.proposals * slots;
    let selector_hidden = weight(
        graph,
        load,
        WeightScope::Draft,
        WeightKind::SelectorHidden,
        weights,
    )?;
    let predecessor_codes = weight(
        graph,
        load,
        WeightScope::Draft,
        WeightKind::SelectorPredecessor,
        weights,
    )?;
    let successor_codes = weight(
        graph,
        load,
        WeightScope::Draft,
        WeightKind::SelectorSuccessor,
        weights,
    )?;
    let top_k_dims = [("M", outputs), ("V", readout_vocabulary), ("K", top_k)];
    let mut candidates = graph.local_for(entries.top_k, "candidates", &top_k_dims)?;
    let mut unary = graph.local_for(entries.top_k, "unary", &top_k_dims)?;
    graph.enqueue(
        entries.top_k,
        &top_k_dims,
        draft_top_k::WorkflowArgs {
            logits: path.logits.into(),
            candidates: candidates.tensor_mut().into(),
            unary: unary.tensor_mut().into(),
        },
    )?;
    let candidates = candidates.tensor().clone();
    let unary = unary.tensor().clone();
    let features = graph
        .enqueue(
            entries.features,
            &[("M", path.rows), ("O", outputs), ("D", hidden)],
            readout_features_rows::WorkflowArgs {
                hidden: path.residual.into(),
                norm: path.output_norm.into(),
                out_rows: path.head_rows.into(),
                epsilon: path.epsilon,
            },
        )?
        .value;
    let projected = graph
        .enqueue(
            entries.hidden,
            &[("M", outputs), ("K", hidden), ("N", rank), ("WS", 0)],
            project_rows::WorkflowArgs {
                source: (&features).into(),
                weight: (&selector_hidden).into(),
                weight_scale: absent_scale.into(),
            },
        )?
        .value;
    let successors = graph
        .enqueue(
            entries.successor,
            &[("M", outputs * top_k), ("V", vocabulary), ("D", rank)],
            embedding_rows::WorkflowArgs {
                table: (&successor_codes).into(),
                tokens: (&candidates).into(),
                scale: 1.0,
                normalize: 0,
                epsilon: path.epsilon,
            },
        )?
        .r0;
    let code_dims = [("M", slots), ("V", vocabulary), ("D", rank)];
    let anchors = graph.input_for(entries.predecessor, "tokens", &code_dims)?;
    for step in 0..path.proposals {
        let (first, last) = (step * slots, (step + 1) * slots);
        let previous = match step {
            0 => anchors.tensor().slice_leading(0, slots),
            _ => path.result.slice_leading(first - slots, first),
        };
        let predecessors = graph
            .enqueue(
                entries.predecessor,
                &code_dims,
                embedding_rows::WorkflowArgs {
                    table: (&predecessor_codes).into(),
                    tokens: (&previous).into(),
                    scale: 1.0,
                    normalize: 0,
                    epsilon: path.epsilon,
                },
            )?
            .r0;
        let mut selection = path.result.slice_leading(first, last);
        graph.enqueue(
            entries.path,
            &[("S", slots), ("K", top_k), ("R", rank)],
            draft_path_step::WorkflowArgs {
                candidates: (&candidates.slice_leading(first * top_k, last * top_k)).into(),
                unary: (&unary.slice_leading(first, last)).into(),
                hidden: (&projected.slice_leading(first, last)).into(),
                predecessor: (&predecessors).into(),
                successor: (&successors.slice_leading(first * top_k, last * top_k)).into(),
                selection: (&mut selection).into(),
            },
        )?;
    }
    Ok(anchors)
}

type PreparedDraftGraph = DraftGraphParts<NativeGraphPlan>;

pub struct PreparedDraftGraphs {
    classes: BTreeMap<DraftGraphClass, PreparedDraftGraph>,
    family: NativeGraphFamily,
    /// The block pass's visible spans.
    segments: u64,
    /// Proposals every drafting class drafts.
    proposals: u64,
}

pub(crate) struct BoundDraftGraphs {
    prepared: Rc<PreparedDraftGraphs>,
    bound: BTreeMap<DraftGraphClass, BoundNativeGraphPlan>,
    constants: Vec<Tensor>,
}

impl PreparedDraftGraphs {
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        distinct_storage_bytes(
            self.classes
                .values()
                .flat_map(|graph| graph.constants.iter()),
        )
    }

    pub(crate) fn prepare(
        target_device: &Device,
        draft: &AttestedDraft,
        target: &AttestedTarget,
        load: &ModelLoadPlan,
        geometry: &DraftGeometry<'_>,
        classes: impl IntoIterator<Item = DraftGraphClass>,
    ) -> Result<Self, SubmitError> {
        let mut prepared = BTreeMap::new();
        for class in classes {
            if prepared.contains_key(&class) {
                return Err(invalid("draft graph class is duplicated"));
            }
            let class_error =
                |error: GraphError| invalid(format!("draft graph class {class:?}: {error}"));
            let parts = draft_graph(
                target_device.native_graph(),
                DraftGraphEntries::prepared(draft, target),
                load,
                geometry,
                class,
            )
            .map_err(class_error)?;
            let plan = GraphDraft::seal(parts.plan).map_err(class_error)?;
            prepared.insert(
                class,
                DraftGraphParts {
                    plan,
                    conditioning: parts.conditioning,
                    injection: parts.injection,
                    block: parts.block,
                    constants: parts.constants,
                    weights: parts.weights,
                    output: parts.output,
                },
            );
        }
        if prepared.is_empty() {
            return Err(invalid("draft graph family has no classes"));
        }
        let plans = prepared
            .values()
            .map(|graph| graph.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self {
            classes: prepared,
            family,
            segments: geometry.segments,
            proposals: geometry.proposals,
        })
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        resident: &ResidentHead,
    ) -> Result<BoundDraftGraphs, SubmitError> {
        let mut uploaded = ConstantTensors::new(resident.embedding.tensor().device());
        let mut bound = BTreeMap::new();
        for (class, graph) in &self.classes {
            let constants = graph
                .constants
                .iter()
                .map(|constant| Ok((constant.port(), uploaded.tensor(constant).map_err(device)?)))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            // Readout-vocabulary weights bind their leading rows.
            let leading = graph
                .block
                .iter()
                .flat_map(|block| {
                    block.leading.iter().map(|(role, port)| {
                        let tensor = resident_draft_weight(resident, *role)?
                            .tensor()
                            .slice_leading(0, block.readout_vocabulary)
                            .map_err(device)?;
                        Ok((port, tensor))
                    })
                })
                .collect::<Result<Vec<_>, SubmitError>>()?;
            let fixed = graph
                .weights
                .iter()
                .map(|(weight, port)| {
                    let resident = resident_draft_weight(resident, weight.role)?;
                    Ok((port, weight.part.of(resident).map_err(invalid)?))
                })
                .chain(constants.iter().map(|(port, tensor)| Ok((*port, tensor))))
                .chain(leading.iter().map(|(port, tensor)| Ok((*port, tensor))))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            bound.insert(*class, graph.plan.bind_static(&fixed).map_err(device)?);
        }
        Ok(BoundDraftGraphs {
            prepared: self.clone(),
            bound,
            constants: uploaded.into_tensors(),
        })
    }
}

impl BoundDraftGraphs {
    fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.constants.iter().try_fold(0u64, |bytes, tensor| {
            bytes
                .checked_add(tensor.storage_bytes())
                .ok_or("draft graph constant charge overflows")
        })
    }

    fn class(
        &self,
        class: DraftGraphClass,
    ) -> Result<(&PreparedDraftGraph, &BoundNativeGraphPlan), SubmitError> {
        let graph = self
            .prepared
            .classes
            .get(&class)
            .ok_or_else(|| invalid(format!("draft graph class {class:?} was not prepared")))?;
        let bound = self
            .bound
            .get(&class)
            .ok_or_else(|| invalid(format!("draft graph class {class:?} was not bound")))?;
        Ok((graph, bound))
    }
}

/// A draft graph's weight: the block's embedding table (the target's or the
/// draft's own), the target's projection, or a drafter weight.
fn resident_draft_weight(
    resident: &ResidentHead,
    role: WeightRole,
) -> Result<&ResidentWeight, SubmitError> {
    match (role.scope, role.kind) {
        (WeightScope::Target | WeightScope::Draft, WeightKind::Embedding) => {
            Ok(&resident.embedding)
        }
        (WeightScope::Target, WeightKind::Output) => Ok(&resident.output),
        _ => resident.weights.get(role).map_err(invalid),
    }
}

/// One pass's attention controls for one layer's history domain, padded to
/// `rows` rows and `segments` spans: padding rows attend to nothing and
/// append nowhere. A block pass (`block` is the layer's block attention)
/// reads its history and its slot's block fresh, through the reading row
/// when causal and wholly when bidirectional; the entry pass reads nothing.
struct PassControls {
    coordinates: Vec<u8>,
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
}

impl PassControls {
    fn new(
        pass: &TargetBatchUpload<'_>,
        domain: usize,
        rows: usize,
        segments: usize,
        block: Option<BlockAttention>,
    ) -> Result<Self, SubmitError> {
        let actual = pass.actual_rows;
        if actual > rows {
            return Err(invalid("draft pass has more rows than its graph class"));
        }
        let history = pass
            .histories
            .get(domain)
            .ok_or_else(|| invalid("draft pass lacks a layer's history domain"))?;
        let mut visible = vec![0_i32; rows * segments * 2];
        // Entry rows only append their K/V. Their attention result does not
        // feed the block pass, so scanning the existing history here would
        // repeat a full-context decode for each draft layer.
        if block.is_some() {
            for (row, ranges) in history.visible[..actual].iter().enumerate() {
                let used = ranges
                    .iter()
                    .rposition(|range| range[1] > range[0])
                    .map_or(0, |last| last + 1);
                if used > segments {
                    return Err(invalid(format!(
                        "draft row attends {used} history spans; the draft admits {segments}"
                    )));
                }
                for (span, [start, end]) in ranges[..used].iter().enumerate() {
                    let at = (row * segments + span) * 2;
                    visible[at] = *start;
                    visible[at + 1] = *end;
                }
            }
        }
        let fresh = (0..rows).map(|row| {
            if row >= actual {
                return [0, 0];
            }
            match block {
                Some(BlockAttention::Causal) => history.fresh[row],
                Some(BlockAttention::Bidirectional) => pass.segments[pass.row_slots[row] as usize],
                None => [0, 0],
            }
        });
        Ok(Self {
            coordinates: i32_bytes((0..rows).flat_map(|row| {
                pass.coordinates
                    .get(row)
                    .filter(|_| row < actual)
                    .copied()
                    .unwrap_or([0; 4])
            })),
            visible: i32_bytes(visible),
            fresh: i32_bytes(fresh.flatten()),
            destinations: i32_bytes((0..rows).map(|row| {
                if row < actual {
                    history.destinations[row]
                } else {
                    -1
                }
            })),
        })
    }
}

/// Widen activation rows (bf16 or f16) to F32.
fn widen(bytes: &[u8], dtype: ActivationDType) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .flat_map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let value = match dtype {
                ActivationDType::BF16 => f32::from_bits(u32::from(bits) << 16),
                ActivationDType::F16 => f16_to_f32(bits),
            };
            value.to_le_bytes()
        })
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

pub struct NativeDraftProgram {
    definition: ModelDefinition,
    graphs: BoundDraftGraphs,
    waiter: CompletionWaiter,
}

impl NativeDraftProgram {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    pub(crate) fn new(
        definition: ModelDefinition,
        graphs: BoundDraftGraphs,
    ) -> Result<Self, SubmitError> {
        let waiter = CompletionWaiter::spawn().map_err(device)?;
        Ok(Self {
            definition,
            graphs,
            waiter,
        })
    }

    fn queue(
        &self,
        core: &HeadLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: NativeGraphOutputLease,
    ) -> Result<(seismic::NativeGraphCompletion, Option<GraphOutputTensor>), SubmitError> {
        let decoder = &self.definition.decoder;
        let draft = self
            .definition
            .draft
            .as_ref()
            .ok_or_else(|| invalid("the draft program's definition has no draft"))?;
        let batch = core.batch();
        let entry = batch.upload();
        let block = batch.chain().next();
        let steps = batch.steps();
        let actual_slots = batch.actual_slots();
        let entry_rows = entry.class.rows();
        let (slots, shaped) = match &block {
            Some(block) => (
                row_class(actual_slots).ok_or_else(|| invalid("draft slots have no class"))?,
                block.shaping[..block.select_rows.len()].iter().any(shapes),
            ),
            None => (0, false),
        };
        let class = DraftGraphClass {
            entry_rows: entry_rows as u64,
            slots: slots as u64,
            shaped,
            windowed: block.is_some() || core.windowed(),
            device: matches!(core.conditioning(), crate::HeadConditioning::Features(_)),
        };
        let (graph, bound) = self.graphs.class(class)?;
        let planes = core
            .advances()
            .first()
            .ok_or_else(|| invalid("draft batch has no state advance"))?
            .bindings()
            .history;
        // Every pass binds each layer's planes, in plane-descriptor order.
        let mut bindings = bound.bindings();
        let mut layer_domains = Vec::with_capacity(draft.blocks.len());
        for layer in 0..draft.blocks.len() {
            let layer_ref = LayerRef::Head(layer as u32);
            let layer_planes = planes
                .iter()
                .filter(|plane| plane.layer == layer_ref)
                .collect::<Vec<_>>();
            let domain = layer_planes
                .first()
                .ok_or_else(|| invalid("a draft layer has no history plane"))?
                .domain
                .0;
            layer_domains.push(domain);
            let passes = graph.injection[layer]
                .iter()
                .chain(graph.block.iter().map(|block| &block.layers[layer]));
            for ports in passes {
                if ports.planes.len() != layer_planes.len() {
                    return Err(invalid(
                        "draft history planes differ from the attention entry",
                    ));
                }
                for (port, plane) in ports.planes.iter().zip(&layer_planes) {
                    bindings.set(port, &plane.buffer).map_err(device)?;
                }
            }
        }
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        match (&graph.conditioning, core.conditioning()) {
            // Host rows in entry-row order, widened to F32 and padded.
            (DraftConditioning::Rows(port), crate::HeadConditioning::Rows(rows)) => {
                let mut conditioning = Vec::new();
                for rows in rows {
                    conditioning.extend(widen(rows.bytes(), decoder.activation_dtype));
                }
                let row_bytes = decoder.hidden as usize * 4;
                if conditioning.len() != entry.actual_rows * row_bytes {
                    return Err(invalid(
                        "draft conditioning bytes differ from the entry rows",
                    ));
                }
                conditioning.resize(entry_rows * row_bytes, 0);
                active.write_input(port, &conditioning).map_err(device)?;
            }
            // The target features' leading class rows; rows past the entry
            // rows condition padding rows, whose injection writes nothing.
            (DraftConditioning::Features(port), crate::HeadConditioning::Features(features)) => {
                let features = features
                    .tensor()
                    .slice_leading(0, entry_rows as u64)
                    .map_err(|error| invalid(format!("draft conditioning features: {error}")))?;
                bindings.set(port, &features).map_err(device)?;
            }
            _ => return Err(invalid("draft conditioning differs from its graph class")),
        }
        for (ports, &domain) in graph.injection.iter().zip(&layer_domains) {
            let Some(ports) = ports else { continue };
            let controls = PassControls::new(&entry, domain, entry_rows, 1, None)?;
            write_controls(&mut active, &ports.controls, &controls)?;
        }
        if let (Some(ports), Some(block)) = (&graph.block, &block) {
            if steps as u64 != self.graphs.prepared.proposals {
                return Err(invalid(
                    "a draft batch drafts other than the load's proposals",
                ));
            }
            let block_rows = slots * draft.block_rows(steps as u64) as usize;
            let segments = self.graphs.prepared.segments as usize;
            active
                .write_input(
                    &ports.tokens,
                    &i32_bytes((0..block_rows).flat_map(|row| {
                        [
                            block
                                .tokens
                                .get(row)
                                .copied()
                                .filter(|_| row < block.actual_rows)
                                .unwrap_or(0),
                            0,
                        ]
                    })),
                )
                .map_err(device)?;
            for ((layer, &domain), &attention) in ports
                .layers
                .iter()
                .zip(&layer_domains)
                .zip(&draft.block_attention)
            {
                let controls =
                    PassControls::new(block, domain, block_rows, segments, Some(attention))?;
                write_controls(&mut active, &layer.controls, &controls)?;
            }
            // Proposal k of slot s: graph row k · slots + s, packed
            // selection s · steps + k; padding slots repeat selection 0.
            let packed = |step: usize, slot: usize| {
                if slot < actual_slots {
                    slot * steps + step
                } else {
                    0
                }
            };
            let head_rows = (0..steps).flat_map(|step| (0..slots).map(move |slot| (step, slot)));
            let rows = head_rows
                .map(|(step, slot)| {
                    let index = packed(step, slot);
                    let output = *block
                        .select_rows
                        .get(index)
                        .ok_or_else(|| invalid("draft selection is absent"))?;
                    block
                        .out_rows
                        .get(output as usize)
                        .copied()
                        .ok_or_else(|| invalid("draft selection has no block row"))
                })
                .collect::<Result<Vec<i32>, SubmitError>>()?;
            active
                .write_input(&ports.head_rows, &i32_bytes(rows))
                .map_err(device)?;
            let words = ports.readout_vocabulary.div_ceil(32) as usize;
            match ports.selections.as_slice() {
                [selection] => {
                    let order = (0..steps)
                        .flat_map(|step| (0..slots).map(move |slot| (step, slot)))
                        .map(|(step, slot)| packed(step, slot))
                        .collect::<Vec<_>>();
                    write_selection_rows(block, &mut active, selection, &order, words)?;
                }
                per_step => {
                    for (step, selection) in per_step.iter().enumerate() {
                        let order = (0..slots)
                            .map(|slot| packed(step, slot))
                            .collect::<Vec<_>>();
                        write_selection_rows(block, &mut active, selection, &order, words)?;
                    }
                }
            }
            if let Some(anchors) = &ports.anchors {
                let anchors_of = (0..slots).flat_map(|slot| {
                    let row = slot * draft.block_rows(steps as u64) as usize;
                    [
                        block
                            .tokens
                            .get(row)
                            .copied()
                            .filter(|_| slot < actual_slots)
                            .unwrap_or(0),
                        0,
                    ]
                });
                active
                    .write_input(anchors, &i32_bytes(anchors_of))
                    .map_err(device)?;
            }
        }
        let mut output = output;
        let outputs = output
            .activate(&graph.plan)
            .map_err(SubmitError::Invariant)?;
        let (outputs, completion) = active
            .attach(bindings, outputs)
            .and_then(|ready| ready.submit())
            .map_err(device)?;
        let owner = output.publish(outputs);
        let selections = (steps > 0)
            .then(|| {
                owner
                    .tensor(&graph.output)
                    .ok_or_else(|| invalid("draft graph omitted its selections"))
            })
            .transpose()?;
        Ok((completion, selections))
    }
}

fn write_controls(
    active: &mut seismic::NativeGraphFamilyActive<'_>,
    ports: &AttentionControlPorts,
    controls: &PassControls,
) -> Result<(), SubmitError> {
    active
        .write_input(&ports.coordinates, &controls.coordinates)
        .map_err(device)?;
    active
        .write_input(&ports.visible, &controls.visible)
        .map_err(device)?;
    active
        .write_input(&ports.fresh, &controls.fresh)
        .map_err(device)?;
    active
        .write_input(&ports.destinations, &controls.destinations)
        .map_err(device)
}

impl HeadProgram for NativeDraftProgram {
    type Submission =
        DeviceSubmission<HeadLaunchCore, NativeGraphWorkspaceLease, Option<GraphOutputTensor>>;

    fn submit(
        &mut self,
        mut launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)> {
        let queued = {
            let (core, workspace, output) = launch.execution_parts_mut();
            match output.take() {
                Some(output) => self.queue(core, workspace, output),
                None => Err(invalid("draft graph output lease is absent")),
            }
        };
        let (completion, selections) = match queued {
            Ok(queued) => queued,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace, _) = launch.into_submission_parts();
        Ok(DeviceSubmission::new(
            self.waiter.completion(vec![completion]),
            core,
            workspace,
            selections,
        ))
    }
}
