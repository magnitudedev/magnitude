//! The native separate-draft program (DFlash, DSpark, DFlash2, LiLiCorr;
//! plan §3.8). One sealed
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
//!   `k + 1`. The service uses a zero confidence threshold, so no confidence
//!   evaluation or additional normalized-feature extraction is needed.
//! - **Confidence.** A method that samples its proposals (DFlash, DSpark)
//!   publishes each one's probability under the logits it sampled it from
//!   (`readout::select`), beside the selections: the host verifies a block
//!   by it. DFlash2 and LiLiCorr choose by path scores, which give no
//!   probability, and publish none.
//!
//! Selections are step-major: proposal `k` of slot `s` is result row
//! `k · slots + s`.

use super::{DeviceSubmission, HeadOutput, HeadProgram, HeadSelections};
use crate::device_table::SelectedRead;
use crate::operators::{self, InputForm, Join, Mixer, SublayerInput};
use crate::{
    completion::CompletionWaiter,
    native::{AttestedDraft, AttestedLanding},
    operators::attention::graph::{
        self as attention_graph, attention_weights, AttentionControlPorts, AttentionOperands,
        AttentionGraphEntries, AttentionHistoryEntries, AttentionWeights, CheckedAttentionEntries, ProjectedRows,
    },
    operators::attention::split::{split, AttentionBlock, AttentionSite, AttentionTopology},
    operators::dense_ffn::graph::{self as dense_graph, CheckedDenseEntries, DenseGraphEntries, DenseNumbers},
    programs::{
        graph::facts::{constant, constant_reals, FactIndex, FactSink, Facts, Numbers, RealIndex, TableIndex},
        graph::readout::{self, shapes, write_selection_rows, SelectionPorts},
        graph::{draft::{Certified, GraphDraft}, GraphError, RowForm},
        native_constants::{
            distinct_storage_bytes, CheckedGraphFamilyResources, CheckedGraphResources,
            ConstantTensors, GraphConstant, RecordedConstant,
        },
        native_target_graph::{
            weight, weight_scale_port, CheckedLandingEntries, GatheredTable, GraphWeights, LandedReadPorts,
            LandingEntries, PackingBound, PlannedWeights, RowDirectory, WeightOperands, WeightPort,
        },
    },
    DeviceError, DraftProgramPlan, HeadLaunchCore, InvariantError,
    ModelLoadPlan, NativeGraphOutputLease, NativeGraphWorkspaceLease, ResidentHead, ResidentOutput,
    ResidentWeight, ResourceLimits, StateStorePlan, SubmitError, ValidatedHeadLaunch,
};
use magnitude_batching::{row_class, TargetBatchUpload};
use magnitude_family_contracts::{
    ActivationDType, AffinePart, BlockAttention, CorrectorSite, CorrectorWeight, DraftDefinition, DraftEmbedding,
    DraftMethod, ModelDefinition, ProjectedRows as ProjectedSource, SublayerIndex, TokenDomain, WeightKind,
    WeightRole, WeightScope,
};
use magnitude_kernels::{
    corrector_attention, corrector_factors, corrector_features, corrector_nodes, corrector_path,
    corrector_rows, dense_output, draft_convolve_input, draft_convolve_residual, draft_gated_rows, draft_path_step,
    draft_top_k, embedding_rows, project_rows, readout_features_rows, sample_rows, shape_rows,
    token_confidence, widen_rows,
};
use magnitude_state::LayerRef;
use seismic::{
    Argument, BackendName, BoundNativeGraphPlan, ClassFact, Device, Dim, Element, NativeGraph, NativeGraphClassSlice,
    NativeGraphFamily, NativeGraphLayout, NativeGraphMetadata, NativeGraphPlan, NativeGraphResourceTemplate,
    NativeGraphTraces, NativePort, Reals, Table, Tensor, WorkflowTensor,
};
use std::{collections::BTreeMap, rc::Rc, sync::Arc};

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

/// One draft graph: the entry rows' class, and when drafting the class of
/// the drafting slots and whether selection is shaped first. Every drafting
/// class records its exact physical proposal count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DraftGraphClass {
    pub entry_rows: u64,
    /// 0 for an injection-only transaction.
    pub slots: u64,
    pub proposals: u64,
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

/// Per draft layer, whether an injection may skip it: a windowed layer
/// narrower than the draft's widest history, which keeps reading the rows
/// the narrower window drops. Layers as wide as the widest are never
/// skipped, so a draft whose layers share one reach injects every layer.
pub(crate) fn skippable_layers(draft: &DraftDefinition) -> Result<Vec<bool>, String> {
    // A layer's reach: its window, or `None` for a history of every row.
    let reaches = (0..draft.blocks.len())
        .map(|index| {
            let paired =
                operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
            let Mixer::Attention(attention) = paired.mixer else {
                return Err("draft layers attend".to_owned());
            };
            Ok(match operators::attention::fresh_rows(attention) {
                operators::attention::FreshRows::Own(ProjectedSource {
                    domain: TokenDomain::Window { tokens },
                    ..
                }) => Some(*tokens),
                _ => None,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let widest = if reaches.contains(&None) {
        None
    } else {
        reaches.iter().flatten().max().copied()
    };
    Ok(reaches
        .iter()
        .map(|reach| reach.is_some_and(|tokens| widest.is_none_or(|widest| tokens < widest)))
        .collect())
}

/// Whether an injection can skip some of the draft's layers
/// ([`skippable_layers`]).
pub(crate) fn skips_windowed_layers(draft: &DraftDefinition) -> Result<bool, String> {
    Ok(skippable_layers(draft)?.contains(&true))
}

/// The admitted draft classes: every entry row class injecting only, and
/// every slot class whose blocks fit a launch drafting one of the
/// `generated` proposal counts after entry rows up to its proposal rows. A
/// load generating none admits no drafting class.
pub(crate) fn draft_graph_classes(
    limits: ResourceLimits,
    proposals: usize,
    generated: &[u8],
    draft: &DraftDefinition,
) -> Result<Vec<DraftGraphClass>, String> {
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
    // Drafting classes project the draft vocabulary per proposing row: their
    // slots are bounded by the selection bound.
    let slot_classes = magnitude_batching::row_classes(limits.max_drafting_slots)
        .into_iter()
        .map(|slots| slots as u64)
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
                    proposals: 0,
                    shaped: false,
                    windowed,
                    device,
                })
        })
        .collect::<Vec<_>>();
    for &generated in generated {
        for &slots in slot_classes
            .iter()
            .filter(|&&slots| slots * draft.block_rows(generated as u64) <= max_rows)
        {
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
                        proposals: generated as u64,
                        shaped,
                        windowed: true,
                        device: false,
                    });
                }
            }
        }
    }
    Ok(classes)
}

/// The token table the draft's block embeds tokens of: the target's, or the
/// draft's own.
pub(crate) fn token_table(draft: &DraftDefinition) -> WeightRole {
    WeightRole {
        scope: match &draft.embedding {
            DraftEmbedding::Target => WeightScope::Target,
            DraftEmbedding::Own(_) => WeightScope::Draft,
        },
        kind: WeightKind::Embedding,
    }
}

/// Every class's fixed geometry: the draft, each layer's history rows and
/// slab rows, the block pass's visible spans, and the proposals drafted.
pub(crate) struct DraftGeometry<'d> {
    pub definition: &'d ModelDefinition,
    pub draft: &'d DraftDefinition,
    /// Per draft layer, its resolved history rows and slab rows.
    pub layers: Vec<(u64, u32)>,
    pub segments: u64,
}

impl<'d> DraftGeometry<'d> {
    pub(crate) fn new(
        definition: &'d ModelDefinition,
        state: &StateStorePlan,
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
        })
    }
}

/// The draft's entries in the graph's binding form.
pub(crate) struct DraftGraphEntries<'a, G: GraphDraft + 'a> {
    pub layers: Vec<DraftLayerEntries<'a, G>>,
    pub embedding: G::Binding<'a, embedding_rows::Entry>,
    /// The landing entries of the block's token table, when it has landing
    /// slots.
    pub landing: Option<LandingEntries<'a, G>>,
    pub head: readout::ProjectionEntries<'a, G>,
    pub shape: G::Binding<'a, shape_rows::Entry>,
    pub sample: G::Binding<'a, sample_rows::Entry>,
    /// Each proposal's probability under the readout logits it was selected
    /// from.
    pub confidence: G::Binding<'a, token_confidence::Entry>,
    /// Widens a device-conditioned entry's target features to F32.
    pub widen: G::Binding<'a, widen_rows::Entry>,
    pub markov: Option<MarkovEntries<'a, G>>,
    pub convolved: Option<ConvolvedEntries<'a, G>>,
    pub selector: Option<SelectorEntries<'a, G>>,
    pub corrector: Option<CorrectorEntries<'a, G>>,
}

/// A convolved backbone's unfused layer entries.
pub(crate) struct ConvolvedEntries<'a, G: GraphDraft + 'a> {
    pub layers: Vec<ConvolvedLayerEntries<'a, G>>,
    pub convolve_input: G::Binding<'a, draft_convolve_input::Entry>,
    pub convolve_residual: G::Binding<'a, draft_convolve_residual::Entry>,
    pub gated: G::Binding<'a, draft_gated_rows::Entry>,
}

/// DFlash2's candidate path.
pub(crate) struct SelectorEntries<'a, G: GraphDraft + 'a> {
    pub features: G::Binding<'a, readout_features_rows::Entry>,
    pub hidden: G::Binding<'a, project_rows::Entry>,
    pub top_k: G::Binding<'a, draft_top_k::Entry>,
    pub predecessor: G::Binding<'a, embedding_rows::Entry>,
    pub successor: G::Binding<'a, embedding_rows::Entry>,
    pub path: G::Binding<'a, draft_path_step::Entry>,
}

/// LiLiCorr's corrector, in graph order (`AttestedCorrector`).
pub(crate) struct CorrectorEntries<'a, G: GraphDraft + 'a> {
    pub pass_norm: G::Binding<'a, readout_features_rows::Entry>,
    pub anchor_rows: G::Binding<'a, readout_features_rows::Entry>,
    pub top_k: G::Binding<'a, draft_top_k::Entry>,
    pub features: G::Binding<'a, corrector_features::Entry>,
    pub feature_down: G::Binding<'a, project_rows::Entry>,
    pub token: Option<G::Binding<'a, project_rows::Entry>>,
    pub pass: G::Binding<'a, project_rows::Entry>,
    pub nodes: G::Binding<'a, corrector_nodes::Entry>,
    pub context: G::Binding<'a, project_rows::Entry>,
    pub context_bias: G::Binding<'a, corrector_rows::Entry>,
    pub anchor_norm: G::Binding<'a, readout_features_rows::Entry>,
    pub attention: G::Binding<'a, corrector_attention::Entry>,
    pub layers: Vec<CorrectorLayerEntries<'a, G>>,
    pub output_norm: G::Binding<'a, readout_features_rows::Entry>,
    pub factors: G::Binding<'a, corrector_factors::Entry>,
    pub factor_input: G::Binding<'a, project_rows::Entry>,
    pub factor_activated: G::Binding<'a, corrector_rows::Entry>,
    pub outgoing: G::Binding<'a, project_rows::Entry>,
    pub incoming: G::Binding<'a, project_rows::Entry>,
    pub anchor: G::Binding<'a, project_rows::Entry>,
    pub unit: G::Binding<'a, corrector_rows::Entry>,
    pub path: G::Binding<'a, corrector_path::Entry>,
}

pub(crate) struct CorrectorLayerEntries<'a, G: GraphDraft + 'a> {
    pub attention_norm: G::Binding<'a, readout_features_rows::Entry>,
    pub attention_input: G::Binding<'a, project_rows::Entry>,
    pub attention_output: G::Binding<'a, project_rows::Entry>,
    pub attention_residual: G::Binding<'a, corrector_rows::Entry>,
    pub mlp_norm: G::Binding<'a, readout_features_rows::Entry>,
    pub up: G::Binding<'a, project_rows::Entry>,
    pub activated: G::Binding<'a, corrector_rows::Entry>,
    pub down: G::Binding<'a, project_rows::Entry>,
    pub down_residual: G::Binding<'a, corrector_rows::Entry>,
}

pub(crate) struct ConvolvedLayerEntries<'a, G: GraphDraft + 'a> {
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
}

impl<'a> DraftGraphEntries<'a, NativeGraph> {
    fn prepared(draft: &'a AttestedDraft, landing: Option<&'a AttestedLanding>) -> Self {
        Self {
            landing: landing.map(LandingEntries::native),
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
            head: readout::ProjectionEntries::prepared(&draft.head),
            shape: &draft.shape,
            sample: &draft.sample,
            confidence: &draft.confidence,
            widen: &draft.widen,
            markov: draft.markov.as_ref().map(|markov| MarkovEntries {
                embedding: &markov.embedding,
                projection: &markov.projection,
            }),
            convolved: draft.convolved.as_ref().map(|convolved| ConvolvedEntries {
                layers: convolved
                    .layers
                    .iter()
                    .map(|layer| ConvolvedLayerEntries {
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
                convolve_input: &convolved.convolve_input,
                convolve_residual: &convolved.convolve_residual,
                gated: &convolved.gated,
            }),
            selector: draft.selector.as_ref().map(|selector| SelectorEntries {
                features: &selector.features,
                hidden: &selector.hidden,
                top_k: &selector.top_k,
                predecessor: &selector.predecessor,
                successor: &selector.successor,
                path: &selector.path,
            }),
            corrector: draft.corrector.as_ref().map(|corrector| CorrectorEntries {
                pass_norm: &corrector.pass_norm,
                anchor_rows: &corrector.anchor_rows,
                top_k: &corrector.top_k,
                features: &corrector.features,
                feature_down: &corrector.feature_down,
                token: corrector.token.as_ref(),
                pass: &corrector.pass,
                nodes: &corrector.nodes,
                context: &corrector.context,
                context_bias: &corrector.context_bias,
                anchor_norm: &corrector.anchor_norm,
                attention: &corrector.attention,
                layers: corrector
                    .layers
                    .iter()
                    .map(|layer| CorrectorLayerEntries {
                        attention_norm: &layer.attention_norm,
                        attention_input: &layer.attention_input,
                        attention_output: &layer.attention_output,
                        attention_residual: &layer.attention_residual,
                        mlp_norm: &layer.mlp_norm,
                        up: &layer.up,
                        activated: &layer.activated,
                        down: &layer.down,
                        down_residual: &layer.down_residual,
                    })
                    .collect(),
                output_norm: &corrector.output_norm,
                factors: &corrector.factors,
                factor_input: &corrector.factor_input,
                factor_activated: &corrector.factor_activated,
                outgoing: &corrector.outgoing,
                incoming: &corrector.incoming,
                anchor: &corrector.anchor,
                unit: &corrector.unit,
                path: &corrector.path,
            }),
        }
    }
}

/// The draft's entries as element assignments of its program plan, for the
/// metadata-only (checked) graphs: part of every draft graph's topology key
/// ([`DraftModelTopology`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedDraftEntries {
    layers: Vec<(CheckedAttentionEntries, CheckedAttentionEntries, CheckedDenseEntries)>,
    embedding: Assignments<2>,
    landing: Option<CheckedLandingEntries>,
    head: readout::CheckedProjectionEntries,
    selection: Assignments<0>,
    widen: Assignments<1>,
    markov: Option<CheckedMarkovEntries>,
    convolved: Option<CheckedConvolvedEntries>,
    selector: Option<CheckedSelectorEntries>,
    corrector: Option<CheckedCorrectorEntries>,
}


type Assignments<const N: usize> = [(&'static str, Element); N];

/// A projection's element assignments: activation rows, weight, published.
fn projected(activation: Element, weight: Element, output: Element) -> Assignments<3> {
    [("A", activation), ("W", weight), ("Y", output)]
}

/// A norm's element assignments.
fn normed(norm: Element, activation: Element) -> Assignments<2> {
    [("NW", norm), ("A", activation)]
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedConvolvedEntries {
    layers: Vec<CheckedConvolvedLayer>,
    activation: Assignments<1>,
    none: Assignments<0>,
}

impl CheckedConvolvedEntries {
    fn new(plan: &DraftProgramPlan, binding: &crate::ConvolutionBinding) -> Self {
        let (a, f32) = (plan.activation(), Element::f32());
        Self {
            layers: binding
                .layers
                .iter()
                .map(|layer| CheckedConvolvedLayer {
                    attention_norm: normed(layer.attention_norm, a),
                    attention_coefficients: projected(
                        a,
                        layer.attention_coefficients.element,
                        f32,
                    ),
                    query: projected(a, layer.query.element, a),
                    key: projected(a, layer.key.element, a),
                    value: projected(a, layer.value.element, a),
                    output: projected(a, layer.output.element, f32),
                    feed_forward_norm: normed(layer.feed_forward_norm, a),
                    feed_forward_coefficients: projected(
                        a,
                        layer.feed_forward_coefficients.element,
                        f32,
                    ),
                    gate: projected(a, layer.gate.element, a),
                    up: projected(a, layer.up.element, a),
                    down: projected(a, layer.down.element, f32),
                })
                .collect(),
            activation: [("A", a)],
            none: [],
        }
    }

    fn entries(&self) -> ConvolvedEntries<'_, NativeGraphMetadata> {
        ConvolvedEntries {
            layers: self
                .layers
                .iter()
                .map(|layer| ConvolvedLayerEntries {
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
            convolve_input: &self.activation[..],
            convolve_residual: &self.none[..],
            gated: &self.activation[..],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedSelectorEntries {
    features: Assignments<2>,
    hidden: Assignments<3>,
    activation: Assignments<1>,
    none: Assignments<0>,
    predecessor: Assignments<2>,
    successor: Assignments<2>,
}

impl CheckedSelectorEntries {
    fn new(plan: &DraftProgramPlan, binding: crate::SelectorBinding) -> Self {
        let a = plan.activation();
        Self {
            features: normed(plan.output_norm(), a),
            hidden: projected(a, binding.hidden, a),
            activation: [("A", a)],
            none: [],
            predecessor: [("EW", binding.predecessor), ("A", a)],
            successor: [("EW", binding.successor), ("A", a)],
        }
    }

    fn entries(&self) -> SelectorEntries<'_, NativeGraphMetadata> {
        SelectorEntries {
            features: &self.features[..],
            hidden: &self.hidden[..],
            top_k: &self.none[..],
            predecessor: &self.predecessor[..],
            successor: &self.successor[..],
            path: &self.activation[..],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedCorrectorEntries {
    pass_norm: Assignments<2>,
    anchor_rows: Assignments<2>,
    feature_down: Assignments<3>,
    token: Option<Assignments<3>>,
    pass: Assignments<3>,
    context: Assignments<3>,
    anchor_norm: Assignments<2>,
    layers: Vec<CheckedCorrectorLayer>,
    output_norm: Assignments<2>,
    factor_input: Assignments<3>,
    outgoing: Assignments<3>,
    incoming: Assignments<3>,
    anchor: Assignments<3>,
    f32_rows: Assignments<1>,
    activated_rows: Assignments<1>,
    activation: Assignments<1>,
    none: Assignments<0>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedCorrectorLayer {
    attention_norm: Assignments<2>,
    attention_input: Assignments<3>,
    attention_output: Assignments<3>,
    mlp_norm: Assignments<2>,
    up: Assignments<3>,
    down: Assignments<3>,
}

impl CheckedCorrectorEntries {
    fn new(plan: &DraftProgramPlan, binding: &crate::CorrectorBinding) -> Self {
        let (a, f32) = (plan.activation(), Element::f32());
        let map = |weight: crate::ScalableWeight| projected(a, weight.element, f32);
        Self {
            pass_norm: normed(plan.output_norm(), a),
            anchor_rows: normed(plan.fusion_norm(), a),
            feature_down: map(binding.feature_down),
            token: binding.token.map(map),
            pass: map(binding.pass),
            context: map(binding.context),
            anchor_norm: normed(binding.anchor_norm, a),
            layers: binding
                .layers
                .iter()
                .map(|layer| CheckedCorrectorLayer {
                    attention_norm: normed(layer.attention_norm, a),
                    attention_input: map(layer.attention_input),
                    attention_output: map(layer.attention_output),
                    mlp_norm: normed(layer.mlp_norm, a),
                    up: map(layer.up),
                    down: map(layer.down),
                })
                .collect(),
            output_norm: normed(binding.output_norm, a),
            factor_input: map(binding.factor_input),
            outgoing: map(binding.outgoing),
            incoming: map(binding.incoming),
            anchor: map(binding.anchor),
            f32_rows: [("Y", f32)],
            activated_rows: [("Y", a)],
            activation: [("A", a)],
            none: [],
        }
    }

    fn entries(&self) -> CorrectorEntries<'_, NativeGraphMetadata> {
        CorrectorEntries {
            pass_norm: &self.pass_norm[..],
            anchor_rows: &self.anchor_rows[..],
            top_k: &self.none[..],
            features: &self.activation[..],
            feature_down: &self.feature_down[..],
            token: self.token.as_ref().map(|token| &token[..]),
            pass: &self.pass[..],
            nodes: &self.none[..],
            context: &self.context[..],
            context_bias: &self.f32_rows[..],
            anchor_norm: &self.anchor_norm[..],
            attention: &self.activation[..],
            layers: self
                .layers
                .iter()
                .map(|layer| CorrectorLayerEntries {
                    attention_norm: &layer.attention_norm[..],
                    attention_input: &layer.attention_input[..],
                    attention_output: &layer.attention_output[..],
                    attention_residual: &self.f32_rows[..],
                    mlp_norm: &layer.mlp_norm[..],
                    up: &layer.up[..],
                    activated: &self.activated_rows[..],
                    down: &layer.down[..],
                    down_residual: &self.f32_rows[..],
                })
                .collect(),
            output_norm: &self.output_norm[..],
            factors: &self.activation[..],
            factor_input: &self.factor_input[..],
            factor_activated: &self.activated_rows[..],
            outgoing: &self.outgoing[..],
            incoming: &self.incoming[..],
            anchor: &self.anchor[..],
            unit: &self.f32_rows[..],
            path: &self.none[..],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedConvolvedLayer {
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedMarkovEntries {
    embedding: [(&'static str, Element); 2],
    projection: [(&'static str, Element); 2],
}

impl CheckedDraftEntries {
    /// `landing`: of the block's token table, when it has landing slots.
    fn new(plan: &DraftProgramPlan, backend: BackendName, landing: Option<CheckedLandingEntries>) -> Result<Self, String> {
        let activation = plan.activation();
        Ok(Self {
            landing,
            layers: plan
                .blocks()
                .iter()
                .map(|block| {
                    Ok((
                        CheckedAttentionEntries::new(block.attention, backend, false)?,
                        CheckedAttentionEntries::new(block.injection, backend, false)?,
                        CheckedDenseEntries::new(block.feed_forward),
                    ))
                })
                .collect::<Result<_, String>>()?,
            embedding: [("EW", plan.embedding().table), ("A", activation)],
            head: readout::CheckedProjectionEntries::new(plan.projection(), plan.output_norm(), activation, false),
            selection: [],
            widen: [("A", activation)],
            markov: plan.markov().map(|markov| CheckedMarkovEntries {
                embedding: [("EW", markov.embedding), ("A", activation)],
                projection: [("DW", markov.projection), ("A", activation)],
            }),
            convolved: plan.convolutions().map(|binding| CheckedConvolvedEntries::new(plan, binding)),
            selector: plan.selector().map(|binding| CheckedSelectorEntries::new(plan, binding)),
            corrector: plan.corrector().map(|binding| CheckedCorrectorEntries::new(plan, binding)),
        })
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
            landing: self.landing.as_ref().map(CheckedLandingEntries::entries),
            head: self.head.entries(),
            shape: &self.selection[..],
            sample: &self.selection[..],
            confidence: &self.selection[..],
            widen: &self.widen[..],
            markov: self.markov.as_ref().map(|markov| MarkovEntries {
                embedding: &markov.embedding[..],
                projection: &markov.projection[..],
            }),
            convolved: self.convolved.as_ref().map(CheckedConvolvedEntries::entries),
            selector: self.selector.as_ref().map(CheckedSelectorEntries::entries),
            corrector: self.corrector.as_ref().map(CheckedCorrectorEntries::entries),
        })
    }
}

/// The class scope of a draft graph's injection pass, whose rows are the
/// class's entry rows.
const INJECTION: &str = "draft-injection";
/// The class dimension of the injection pass's rows, for the ports it
/// declares with them.
const ENTRY_ROWS: &str = "draft_entry_rows";

/// The model numbers a draft's graphs are built of beside its topology
/// ([`DraftModelTopology`]), the same in every class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DraftModelNumbers<D> {
    /// The decoder's (and the draft's) hidden width.
    hidden: D,
    /// The decoder's vocabulary: the readout's rows and the codebooks'.
    vocabulary: D,
    /// Per draft layer, its dense feed-forward's widths.
    dense: Vec<DenseNumbers<D>>,
    /// A convolved backbone's taps and coefficient group width.
    convolution: Option<[D; 2]>,
    method: MethodNumbers<D>,
    /// The weights the graphs read.
    weights: GraphWeights<D>,
}

/// The method and what its selection reads beside the readout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MethodNumbers<D> {
    DFlash,
    /// DSpark and its Markov table width.
    DSpark { rank: D },
    /// DFlash2, its selector's top-k and its codebook width.
    DFlash2 { top_k: D, rank: D },
    /// LiLiCorr, its corrector's top-k and epsilons, and its widths.
    LiLiCorr { topology: CorrectorTopology<RealIndex>, widths: CorrectorNumbers<D> },
}

/// LiLiCorr's corrector widths.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CorrectorNumbers<D> {
    width: D,
    heads: D,
    factor: D,
    mlp: D,
}

impl<D> Numbers<D> for DraftModelNumbers<D> {
    type Of<E> = DraftModelNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> DraftModelNumbers<E> {
        DraftModelNumbers {
            hidden: f(self.hidden),
            vocabulary: f(self.vocabulary),
            dense: self.dense.into_iter().map(|dense| dense.map(&mut *f)).collect(),
            convolution: self.convolution.map(|pair| pair.map(&mut *f)),
            method: match self.method {
                MethodNumbers::DFlash => MethodNumbers::DFlash,
                MethodNumbers::DSpark { rank } => MethodNumbers::DSpark { rank: f(rank) },
                MethodNumbers::DFlash2 { top_k, rank } => MethodNumbers::DFlash2 { top_k: f(top_k), rank: f(rank) },
                MethodNumbers::LiLiCorr { topology, widths } => MethodNumbers::LiLiCorr {
                    topology,
                    widths: CorrectorNumbers {
                        width: f(widths.width),
                        heads: f(widths.heads),
                        factor: f(widths.factor),
                        mlp: f(widths.mlp),
                    },
                },
            },
            weights: self.weights.map(&mut *f),
        }
    }
}

/// A class's part of its graph's key, at `D` its numbers and `A` its
/// attention topologies (indexed in a key, resolved in a builder): its
/// topology, its entry rows (every injection's rows, the one fact the
/// classes of a topology vary) and its layers' attention in each pass it
/// runs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DraftClass<D, A = AttentionTopology<RealIndex, TableIndex>> {
    topology: ClassTopology,
    entry_rows: D,
    passes: ClassPasses<AttentionPass<D, A>>,
}

impl<D, A> DraftClass<D, A> {
    /// It with each pass's attention topology `f` of its own.
    fn resolve<B>(self, mut f: impl FnMut(A) -> B) -> DraftClass<D, B> {
        DraftClass {
            topology: self.topology,
            entry_rows: self.entry_rows,
            passes: self.passes.map(|pass| AttentionPass { topology: f(pass.topology), numbers: pass.numbers }),
        }
    }
}

/// One layer's attention in one pass as a class's split gives it, before
/// its numbers and reals are a key's or a builder's.
type SplitPass = AttentionPass<u64, AttentionTopology<f64, Arc<[u8]>>>;

/// A class's passes over the draft's layers: an injection-only class
/// injects the layers it does not skip; a drafting class injects every
/// layer and runs each in its block pass.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ClassPasses<A> {
    Injection(Vec<Option<A>>),
    Drafting(Vec<DraftingLayer<A>>),
}

/// One layer's attention in a drafting class's injection and block passes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DraftingLayer<A> {
    injection: A,
    block: A,
}

/// One layer's attention in one pass: its topology (its reals and tables
/// indexed in a key, as arguments and tables in a builder) and its numbers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AttentionPass<D, A = AttentionTopology<RealIndex, TableIndex>> {
    topology: A,
    numbers: AttentionBlock<D>,
}

impl<A> ClassPasses<A> {
    fn map<B>(self, mut f: impl FnMut(A) -> B) -> ClassPasses<B> {
        self.map_passes(|pass, _| f(pass))
    }

    /// It with each pass `f` of its own and whether it is an injection.
    fn map_passes<B>(self, mut f: impl FnMut(A, bool) -> B) -> ClassPasses<B> {
        match self {
            Self::Injection(layers) => {
                ClassPasses::Injection(layers.into_iter().map(|layer| layer.map(|pass| f(pass, true))).collect())
            }
            Self::Drafting(layers) => ClassPasses::Drafting(
                layers
                    .into_iter()
                    .map(|layer| DraftingLayer { injection: f(layer.injection, true), block: f(layer.block, false) })
                    .collect(),
            ),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Injection(layers) => layers.len(),
            Self::Drafting(layers) => layers.len(),
        }
    }

    /// Each layer's injection pass, none where the class skips it.
    fn injection(&self, index: usize) -> Option<&A> {
        match self {
            Self::Injection(layers) => layers[index].as_ref(),
            Self::Drafting(layers) => Some(&layers[index].injection),
        }
    }
}

impl<D, A> Numbers<D> for DraftClass<D, A> {
    type Of<E> = DraftClass<E, A>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> DraftClass<E, A> {
        DraftClass {
            topology: self.topology,
            entry_rows: f(self.entry_rows),
            passes: self
                .passes
                .map(|pass| AttentionPass { topology: pass.topology, numbers: pass.numbers.map(&mut *f) }),
        }
    }
}

/// What decides a draft class's graph beside its model's: the classes a
/// topology's template certifies differ only in their entry rows, whose
/// attention form (`RowForm`) it fixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ClassTopology {
    slots: u64,
    proposals: u64,
    /// Rows of each slot's block (`DraftDefinition::block_rows`).
    block_rows: u64,
    shaped: bool,
    windowed: bool,
    device: bool,
    rows: RowForm,
}

impl ClassTopology {
    fn of(class: DraftGraphClass, draft: &DraftDefinition) -> Self {
        Self {
            slots: class.slots,
            proposals: class.proposals,
            block_rows: draft.block_rows(class.proposals),
            shaped: class.shaped,
            windowed: class.windowed,
            device: class.device,
            rows: RowForm::of(class.entry_rows),
        }
    }
}

/// A corrector's top-k (which sizes its norms' identity constants) and,
/// at `R`, its epsilons: each layer's attention and MLP norms', its
/// feature, output and anchor norms', and its unit vectors'.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CorrectorTopology<R> {
    top_k: u64,
    layers: Vec<[R; 2]>,
    feature: R,
    output: R,
    anchor: R,
    vector: R,
}

impl<R> Numbers<R> for CorrectorTopology<R> {
    type Of<E> = CorrectorTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> CorrectorTopology<E> {
        CorrectorTopology {
            top_k: self.top_k,
            layers: self.layers.into_iter().map(|pair| pair.map(&mut *f)).collect(),
            feature: f(self.feature),
            output: f(self.output),
            anchor: f(self.anchor),
            vector: f(self.vector),
        }
    }
}

/// A draft's own scalars, at `R`: the output norm's epsilon (every draft
/// norm's), the block embedding's scale and the readout's logit softcap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DraftScalars<R> {
    epsilon: R,
    embedding_scale: R,
    softcap: R,
}

impl<R> Numbers<R> for DraftScalars<R> {
    type Of<E> = DraftScalars<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> DraftScalars<E> {
        DraftScalars { epsilon: f(self.epsilon), embedding_scale: f(self.embedding_scale), softcap: f(self.softcap) }
    }
}

/// A draft's topology beside its classes': everything its graphs are built
/// of that no model number is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DraftModelTopology {
    backend: BackendName,
    activation: Element,
    /// The block's mask rows (`DraftDefinition::max_proposals`), which its
    /// constants and views are sized by.
    mask_rows: u64,
    /// Each layer's dense feed-forward expansion.
    expansions: Vec<operators::routed::ExpansionForm>,
    /// Whether the method samples its proposals, publishing each one's
    /// probability (`DraftMethodVariant::samples_proposals`).
    samples: bool,
    /// The draft's element assignments.
    entries: CheckedDraftEntries,
    token_scope: WeightScope,
    /// The vocabulary projection as stored: the packed projection, or the
    /// planes of a scope another reader selected a class of
    /// (`ModelLoadPlan::finalize_readout`), read through their full pass.
    projection: readout::StoredHead,
    scalars: DraftScalars<RealIndex>,
}

/// A draft's topology and its numbers' facts: one per model and backend
/// ([`DraftSplit::of`]), in every class's key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DraftModel {
    topology: DraftModelTopology,
    numbers: DraftModelNumbers<FactIndex>,
}

/// A draft graph's topology key ([`record_draft_graph`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DraftKey {
    model: Arc<DraftModel>,
    class: DraftClass<FactIndex>,
}

/// The weights a draft graph reads: the drafter's, and the target's token
/// table and readout it shares.
fn draft_reads(role: WeightRole) -> bool {
    match role.scope {
        WeightScope::Draft | WeightScope::DraftSublayer(_) | WeightScope::CorrectorLayer(_) => true,
        WeightScope::Target => {
            matches!(role.kind, WeightKind::Embedding | WeightKind::Output | WeightKind::OutputPlane(_))
        }
        _ => false,
    }
}

/// A draft planned on a backend, split once into what every class's graph
/// is built of ([`Self::class`]): the only reading of the model behind a
/// draft graph.
struct DraftSplit<'d> {
    draft: &'d DraftDefinition,
    /// Each layer's history rows and slab rows, and the history's visible
    /// spans (`DraftGeometry`).
    histories: &'d [(u64, u32)],
    segments: u64,
    model: Arc<DraftModel>,
    /// The model numbers' facts, which every class's extend.
    facts: FactSink,
    /// Each layer's attention operator and shape, and whether an injection
    /// may skip it (`skippable_layers`).
    attentions: Vec<(&'d magnitude_family_contracts::Attention, crate::AttentionShape, bool)>,
}

impl<'d> DraftSplit<'d> {
    fn of(
        backend: BackendName,
        load: &ModelLoadPlan,
        plan: &DraftProgramPlan,
        geometry: &'d DraftGeometry<'d>,
    ) -> Result<Self, GraphError> {
        let decoder = &geometry.definition.decoder;
        let draft = geometry.draft;
        if geometry.layers.len() != draft.blocks.len() {
            return Err("draft geometry disagrees with the draft's layers".into());
        }
        let skippable = skippable_layers(draft)?;
        let mut facts = FactSink::new();
        let (mut expansions, mut attentions, mut denses) = (Vec::new(), Vec::new(), Vec::new());
        for index in 0..draft.blocks.len() {
            let paired = operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
            let Mixer::Attention(attention) = paired.mixer else {
                return Err("draft layers attend".into());
            };
            let Some(operators::FeedForward::Dense(dense)) = paired.feed_forward.map(|sublayer| sublayer.op) else {
                return Err("draft layers have a dense feed-forward".into());
            };
            let shape =
                operators::attention::shape(decoder.hidden, attention, None).map_err(|error| error.to_string())?;
            let (expansion, widths) = dense_graph::split(load, draft_sublayer(index, 1)?, &dense.up)?;
            expansions.push(expansion);
            attentions.push((attention, shape, skippable[index]));
            denses.push(widths);
        }
        let mut real = |value: f64| facts.real(value);
        let method = match &draft.method {
            DraftMethod::DFlash => MethodNumbers::DFlash,
            DraftMethod::DSpark { markov, .. } => MethodNumbers::DSpark { rank: markov.rank },
            DraftMethod::DFlash2 { selector } => MethodNumbers::DFlash2 { top_k: selector.top_k, rank: selector.rank },
            DraftMethod::LiLiCorr { corrector } => MethodNumbers::LiLiCorr {
                topology: CorrectorTopology {
                    top_k: corrector.top_k,
                    layers: corrector
                        .layers
                        .iter()
                        .map(|layer| [real(layer.attention_norm.epsilon), real(layer.mlp_norm.epsilon)])
                        .collect(),
                    feature: real(corrector.feature_norm.epsilon),
                    output: real(corrector.output_norm.epsilon),
                    anchor: real(corrector.anchor_norm.epsilon),
                    vector: real(corrector.vector_epsilon),
                },
                widths: CorrectorNumbers {
                    width: corrector.width,
                    heads: corrector.heads,
                    factor: corrector.factor,
                    mlp: corrector.mlp(),
                },
            },
        };
        let numbers = DraftModelNumbers {
            hidden: decoder.hidden,
            vocabulary: decoder.vocabulary,
            dense: denses,
            convolution: draft.convolutions.as_ref().map(|convolutions| [convolutions.kernel, convolutions.group]),
            method,
            weights: GraphWeights::of(load, draft_reads),
        };
        let topology = DraftModelTopology {
            backend,
            activation: activation(decoder.activation_dtype),
            mask_rows: draft.max_proposals(),
            expansions,
            samples: draft.method.variant().samples_proposals(),
            entries: CheckedDraftEntries::new(plan, backend, CheckedLandingEntries::of(load, token_table(draft))?)?,
            token_scope: token_table(draft).scope,
            projection: readout::HeadPlans::of(
                load,
                match draft.output {
                    Some(_) => WeightScope::Draft,
                    None => WeightScope::Target,
                },
            )?
            .selected(),
            scalars: facts.index_reals(DraftScalars {
                epsilon: draft.output_norm.epsilon,
                embedding_scale: draft.embedding_scale,
                softcap: draft.logit_softcap.unwrap_or(0.0),
            }),
        };
        Ok(Self {
            draft,
            histories: &geometry.layers,
            segments: geometry.segments,
            model: Arc::new(DraftModel { numbers: facts.index(numbers), topology }),
            facts,
            attentions,
        })
    }

    /// The attention site of a draft layer's pass over `rows` rows in
    /// `segments` visible spans of its history (`history`: its rows and
    /// slab rows); an injection (`inject_only`) appends the entry rows'
    /// keys and values alone.
    fn site(
        operator: &'d magnitude_family_contracts::Attention,
        shape: crate::AttentionShape,
        (epsilon, activation): (f32, Element),
        rows: u64,
        segments: u64,
        (history_rows, slab_rows): (u64, u32),
        inject_only: bool,
    ) -> AttentionSite<'d> {
        AttentionSite {
            rows,
            segments,
            history_rows,
            slab_rows,
            // The draft's classes list no history row tiles.
            history_tiles: 0,
            shape,
            operator,
            epsilon,
            head_epsilon: epsilon,
            post_norm: None,
            activation,
            inject_only,
            causal_cache: false,
            selection: None,
            picked: None,
            controls: crate::programs::graph::draft::ControlSource::Uploaded,
        }
    }

    /// Class `class`'s topology and each layer's attention in each pass it
    /// runs, split: every injection over the class's entry rows.
    fn split_class(&self, class: DraftGraphClass) -> Result<(ClassTopology, ClassPasses<SplitPass>), GraphError> {
        let topology = ClassTopology::of(class, self.draft);
        let scalars = (self.draft.output_norm.epsilon as f32, self.model.topology.activation);
        let pass = |index: usize, rows: u64, segments: u64, inject_only: bool| -> Result<SplitPass, GraphError> {
            let (operator, shape, _) = self.attentions[index];
            let site = Self::site(operator, shape, scalars, rows, segments, self.histories[index], inject_only);
            let (topology, numbers) = split(&site)?;
            Ok(AttentionPass { topology, numbers })
        };
        let layers = 0..self.attentions.len();
        let passes = match class.slots {
            0 => ClassPasses::Injection(
                layers
                    .map(|index| {
                        let skippable = self.attentions[index].2;
                        (!skippable || class.windowed)
                            .then(|| pass(index, class.entry_rows, 1, true))
                            .transpose()
                    })
                    .collect::<Result<_, _>>()?,
            ),
            slots => ClassPasses::Drafting(
                layers
                    .map(|index| {
                        Ok(DraftingLayer {
                            injection: pass(index, class.entry_rows, 1, true)?,
                            block: pass(index, slots * topology.block_rows, self.segments, false)?,
                        })
                    })
                    .collect::<Result<_, GraphError>>()?,
            ),
        };
        Ok((topology, passes))
    }

    /// Class `class`'s key and facts: the model's, its entry rows and each
    /// layer's attention in each pass the class runs.
    fn class(&self, class: DraftGraphClass) -> Result<(DraftKey, FactSink), GraphError> {
        let (topology, passes) = self.split_class(class)?;
        let mut facts = self.facts.clone();
        let entry_rows = facts.fact(class.entry_rows);
        let passes = passes.map_passes(|pass, injection| {
            let topology = facts.index_reals(pass.topology).tables(|bytes| facts.table(bytes));
            // Every injection's rows are the entry rows' one fact.
            let numbers = match injection {
                true => pass
                    .numbers
                    .map(&mut Some)
                    .at_rows(None)
                    .map(&mut |number: Option<u64>| number.map_or(entry_rows, |number| facts.fact(number))),
                false => facts.index(pass.numbers),
            };
            AttentionPass { topology, numbers }
        });
        let key = DraftKey { model: Arc::clone(&self.model), class: DraftClass { topology, entry_rows, passes } };
        Ok((key, facts))
    }
}

/// The draft graphs' traces, one per topology, each with its graph's
/// binding constants.
static DRAFT_TRACES: NativeGraphTraces<DraftKey, Vec<RecordedConstant>> = NativeGraphTraces::new();

/// How many draft graph traces this process recorded.
#[cfg(test)]
pub(crate) fn draft_traces_recorded() -> usize {
    DRAFT_TRACES.recorded()
}

/// The template of the draft graph of `key` at `facts`, and its binding
/// constants.
fn record_draft_graph(
    key: &DraftKey,
    facts: &[Dim],
    reals: &Reals,
) -> Result<(NativeGraphResourceTemplate, Vec<RecordedConstant>), GraphError> {
    let facts = Facts::new(facts, reals);
    let model = &key.model.topology;
    // Each pass's reals as arguments, its tables as the model's (which the
    // trace reads at every model's own).
    let class = facts
        .dims(key.class.clone())
        .resolve(|topology| facts.arguments(topology).tables(|table| facts.table(table)));
    let parts = draft_graph(
        NativeGraphMetadata::new_template(model.backend),
        model.entries.entries()?,
        model,
        (&facts.dims(key.model.numbers.clone()), &class),
        &facts,
    )?;
    Ok((parts.plan.seal_template()?, parts.constants.iter().map(GraphConstant::recorded).collect()))
}

/// The storage and bound constants of the draft's graph family over
/// `classes`, from metadata alone (`certify_draft_family`).
pub(crate) fn checked_draft_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    plan: &DraftProgramPlan,
    geometry: &DraftGeometry<'_>,
    classes: impl IntoIterator<Item = DraftGraphClass>,
) -> Result<CheckedGraphResources, GraphError> {
    let classes = classes.into_iter().collect::<Vec<_>>();
    let split = DraftSplit::of(backend, load, plan, geometry)?;
    certify_draft_family::<seismic::NativeGraphOutline>(&split, &classes).map(|(resources, _)| resources)
}

/// The family's storage and each class's layout. Every class field but the
/// entry rows fixes a graph's topology ([`ClassTopology`]), as does the
/// attention form those rows take (`RowForm`): only the injection pass
/// follows them, the block pass's rows being its slots' and proposals'. So
/// each topology is one template, recorded at its largest entry rows (whose
/// constants hold every smaller class's) and certified over every class's, and each class seals into its
/// topology's layout (`PreparedDraftGraphs::prepare`). The injection pass's
/// only constants that follow its rows are empty (an attention prefill's
/// `[rows, 0]` bitmap), so the template's constants are every class's.
fn certify_draft_family<L: Certified>(
    split: &DraftSplit<'_>,
    classes: &[DraftGraphClass],
) -> Result<(CheckedGraphResources, BTreeMap<DraftGraphClass, L>), GraphError> {
    let mut groups: BTreeMap<ClassTopology, Vec<DraftGraphClass>> = BTreeMap::new();
    for &class in classes {
        let group = groups.entry(ClassTopology::of(class, split.draft)).or_default();
        if group.contains(&class) {
            return Err("draft graph class is duplicated".into());
        }
        group.push(class);
    }
    if groups.is_empty() {
        return Err("draft graph family has no classes".into());
    }
    let mut family = CheckedGraphFamilyResources::new();
    let mut layouts = BTreeMap::new();
    for (topology, group) in groups {
        let context = || format!("draft graph topology {topology:?}");
        // Recorded at its largest entry rows: the recording's constants are
        // charged for every class it certifies, and the largest's hold
        // every smaller class's.
        let largest = *group.iter().max_by_key(|class| class.entry_rows).expect("a topology holds a class");
        let (key, facts) = split.class(largest)?;
        let slices = group
            .iter()
            .map(|class| {
                NativeGraphClassSlice::new()
                    .dimension(ENTRY_ROWS, [class.entry_rows])
                    .scoped(INJECTION, "M", [class.entry_rows])
            })
            .collect::<Vec<_>>();
        let (layout, constants) = L::traced(
            &DRAFT_TRACES,
            &key,
            &facts,
            &[ClassFact { fact: key.class.entry_rows.position(), scope: None, dimension: ENTRY_ROWS }],
            slices,
            record_draft_graph,
        )
        .map_err(|error| error.context(context()))?;
        family.include(layout.storage_bytes(), constants);
        // A draft launch runs on its own lane, never in a decode step's.
        family.include_graph(layout.storage_bytes().recorded, false);
        for &class in &group {
            family.include_plan(layout.plan_heap())?;
            layouts.insert(class, layout.clone());
        }
    }
    Ok((family.finish()?, layouts))
}

/// The draft family over `classes` as `L` certifies it
/// ([`certify_draft_family`]): the load's layouts, or assessment's outlines.
#[cfg(test)]
pub(crate) fn certified_draft_family<L: Certified>(
    backend: BackendName,
    load: &ModelLoadPlan,
    plan: &DraftProgramPlan,
    geometry: &DraftGeometry<'_>,
    classes: &[DraftGraphClass],
) -> Result<(CheckedGraphResources, BTreeMap<DraftGraphClass, L>), GraphError> {
    certify_draft_family(&DraftSplit::of(backend, load, plan, geometry)?, classes)
}

/// Every class's exact graph seals into the layout it is certified into, at
/// that layout's storage, and holds no nonempty constant the family does
/// not charge.
#[cfg(test)]
pub(crate) fn verify_draft_family_certificates(
    backend: BackendName,
    load: &ModelLoadPlan,
    plan: &DraftProgramPlan,
    geometry: &DraftGeometry<'_>,
    classes: &[DraftGraphClass],
) -> Result<(), GraphError> {
    let split = DraftSplit::of(backend, load, plan, geometry)?;
    let (resources, layouts) = certify_draft_family::<NativeGraphLayout>(&split, classes)?;
    let entries = &split.model.topology.entries;
    let numbers = split.numbers();
    let mut exact = CheckedGraphFamilyResources::new();
    for &class in classes {
        let parts = split.build(NativeGraphMetadata::new(backend), entries.entries()?, &numbers, class)?;
        let charged = parts
            .plan
            .seal_with_layout(&layouts[&class])
            .map_err(|error| GraphError::from(error).context(format!("draft class {class:?}")))?;
        if charged != layouts[&class].storage_bytes() {
            return Err(format!("draft class {class:?} charged a different layout").into());
        }
        exact.include(seismic::NativeGraphStorageBytes::default(), parts.constants);
    }
    let exact = exact.finish()?;
    if exact.binding_constant_bytes != resources.binding_constant_bytes {
        return Err(format!(
            "the draft family charges {} constant bytes, its classes hold {}",
            resources.binding_constant_bytes, exact.binding_constant_bytes
        )
        .into());
    }
    Ok(())
}

impl DraftSplit<'_> {
    /// The model's numbers at its own values, which every class a load
    /// builds reads ([`Self::build`]).
    fn numbers(&self) -> DraftModelNumbers<Dim> {
        let facts = self.facts.values().iter().copied().map(Dim::from).collect::<Vec<_>>();
        Facts::new(&facts, self.facts.reals()).dims(self.model.numbers.clone())
    }

    /// Class `class`'s graph at the model's own `numbers` ([`Self::numbers`])
    /// and the class's, as a load builds it: every number a constant, every
    /// table the model's own bytes.
    fn build<'a, G: GraphDraft + 'a>(
        &self,
        graph: G,
        entries: DraftGraphEntries<'a, G>,
        numbers: &DraftModelNumbers<Dim>,
        class: DraftGraphClass,
    ) -> Result<DraftGraphParts<G>, GraphError> {
        let (topology, passes) = self.split_class(class)?;
        let class = DraftClass {
            topology,
            entry_rows: Dim::from(class.entry_rows),
            passes: passes.map(|pass| AttentionPass {
                topology: constant_reals(pass.topology).tables(Table::from),
                numbers: constant(pass.numbers),
            }),
        };
        let dims = self.facts.values().iter().copied().map(Dim::from).collect::<Vec<_>>();
        draft_graph(graph, entries, &self.model.topology, (numbers, &class), &Facts::new(&dims, self.facts.reals()))
    }
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
    /// The block row of each projected row: each proposal's, step-major, or
    /// (a corrector's) every mask row, slot-major.
    head_rows: NativePort,
    layers: Vec<LayerPorts>,
    /// One selection for every proposal (DFlash), or one per step (DSpark);
    /// none when a selector or corrector selects.
    selections: Vec<SelectionPorts>,
    /// DSpark and DFlash2: each slot's anchor, the token its first step
    /// conditions on.
    anchors: Option<NativePort>,
    /// A corrector's: each slot's last entry row, whose conditioning row is
    /// the fused feature that selected its anchor.
    anchor_rows: Option<NativePort>,
    /// A corrector's landed read of its candidates' token-table rows, of a
    /// table with landing slots.
    landed: Option<LandedReadPorts>,
    /// Weights whose rows are vocabulary entries of the readout (the target's
    /// projection, DSpark's Markov projection), bound to their leading
    /// `readout_vocabulary` rows.
    leading: Vec<(WeightRole, NativePort)>,
    readout_vocabulary: Dim,
    /// `[proposals · slots]`, step-major: each proposal's probability under
    /// the logits it was sampled from; absent for a method choosing by path
    /// scores.
    confidence: Option<WorkflowTensor>,
}

/// A head projection's ports over its leading `vocabulary` rows, kept as
/// `leading` ports, and its scale registered as the weight part it binds
/// (`readout::StoredHead::ports`).
struct VocabularyPrefix<'p> {
    planned: &'p GraphWeights,
    vocabulary: &'p Dim,
    leading: &'p mut Vec<(WeightRole, NativePort)>,
    weights: &'p mut Vec<(WeightPort, NativePort)>,
    constants: &'p mut Vec<GraphConstant>,
}

impl<G: GraphDraft> readout::ProjectionBinding<G> for VocabularyPrefix<'_> {
    fn weight(&mut self, graph: &mut G, role: WeightRole) -> Result<NativePort, GraphError> {
        let plan = self.planned.weight(role)?;
        let mut shape = plan.shape;
        shape[0] = self.vocabulary.clone();
        let port = graph.port(plan.resident, &shape)?;
        self.leading.push((role, port.clone()));
        Ok(port)
    }

    fn scale(&mut self, graph: &mut G, role: WeightRole) -> Result<(NativePort, Dim), GraphError> {
        weight_scale_port(graph, self.planned, role, self.weights, self.constants)
    }
}

/// A port over the leading `rows` rows of `role`'s planned weight, whose
/// rows are `width` wide.
fn leading_port<G: GraphDraft>(
    graph: &mut G,
    planned: &GraphWeights,
    role: WeightRole,
    rows: &Dim,
    width: &Dim,
    leading: &mut Vec<(WeightRole, NativePort)>,
) -> Result<WorkflowTensor, GraphError> {
    let port = graph.port(planned.weight(role)?.resident, &[rows, width])?;
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

fn draft_sublayer(block: usize, sublayer: u32) -> Result<WeightScope, String> {
    Ok(WeightScope::DraftSublayer(SublayerIndex {
        block: u32::try_from(block).map_err(|_| "draft block index exceeds u32")?,
        sublayer,
    }))
}

/// The graph of draft class `draft_class` of the draft `model` at its
/// `numbers`, the model's scalars read as arguments from `reals`: its only
/// inputs beside its entries, so a recording ([`record_draft_graph`]) and a
/// load ([`DraftSplit::build`]) build the same graph. Each pass's attention
/// topology comes resolved: its tables a recording's facts or a load's
/// bytes.
fn draft_graph<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: DraftGraphEntries<'a, G>,
    model: &DraftModelTopology,
    (numbers, draft_class): (&DraftModelNumbers<Dim>, &DraftClass<Dim, AttentionTopology>),
    reals: &Facts<'_>,
) -> Result<DraftGraphParts<G>, GraphError> {
    let class = draft_class.topology;
    let passes = &draft_class.passes;
    let planned = &numbers.weights;
    let (hidden, vocabulary) = (&numbers.hidden, &numbers.vocabulary);
    let activation = model.activation;
    let scalars = reals.arguments(model.scalars);
    let (epsilon, embedding_scale, softcap) = (scalars.epsilon.f32(), scalars.embedding_scale.f32(), scalars.softcap.f32());
    let layer_count = model.expansions.len();
    if [entries.layers.len(), passes.len(), numbers.dense.len()]
        .iter()
        .any(|&count| count != layer_count)
    {
        return Err("draft entries disagree with the draft's layers".into());
    }
    let mut weights = Vec::new();
    let mut constants = Vec::new();
    let fusion_norm = weight(&mut graph, planned, WeightScope::Draft, WeightKind::DraftFusionNorm, &mut weights)?;
    // Each layer's attention weights for injection, whose input norm is the
    // fusion norm; the block pass reads the same weights with the layer's
    // own input norm.
    let mut layer_weights = Vec::with_capacity(layer_count);
    for index in 0..layer_count {
        // An injection-only class skips the layers it does not inject.
        let Some(AttentionPass { topology, numbers }) = passes.injection(index) else {
            layer_weights.push(None);
            continue;
        };
        let scope = draft_sublayer(index, 0)?;
        let operands = AttentionOperands::planned(
            &mut graph,
            planned,
            scope,
            topology,
            numbers,
            class.slots > 0,
            &mut weights,
            &mut constants,
        )?;
        let injection = attention_weights(
            topology,
            numbers,
            InputForm::SINGLE,
            class.slots > 0,
            operands,
            |kind| match kind {
                WeightKind::InputNorm => Ok(fusion_norm.clone()),
                kind => weight(&mut graph, planned, scope, kind, &mut weights),
            },
        )?;
        layer_weights.push(Some(injection));
    }

    // Injection: each entry row appends its context K/V at its destination.
    // The host writes the rows (an input of the first injection's
    // projection; every layer's projection reads the same rows), or a
    // device-conditioned class widens its bound target features.
    let (first, first_weights) = layer_weights
        .iter()
        .enumerate()
        .find_map(|(index, layer)| layer.as_ref().map(|layer| (index, layer)))
        .ok_or("a draft has no injected layers")?;
    let first_pass = passes.injection(first).expect("an injected layer has its injection");
    let entry_rows = &draft_class.entry_rows;
    // Only the injection pass follows the entry rows: its rows are a
    // topology's class dimension (`certify_draft_family`).
    graph.set_class_scope(Some(INJECTION));
    let (conditioning, conditioning_rows) = if class.device {
        let features = graph.port_with_class_extent(activation, &[entry_rows, hidden], 0, ENTRY_ROWS)?;
        let widened = graph
            .enqueue(
                entries.widen,
                &[("M", entry_rows), ("D", hidden)],
                widen_rows::WorkflowArgs { rows: features.tensor().into() },
            )?
            .value;
        (DraftConditioning::Features(features), widened)
    } else {
        let rows = graph.input_for(
            entries.layers[first].injection.project,
            "hidden",
            &[
                first_pass
                    .numbers
                    .widths
                    .key_value_dimensions(&first_pass.topology.form, first_pass.numbers.rows.clone())
                    .as_slice(),
                first_weights.operands.project_dimensions(true).as_slice(),
                &InputForm::SINGLE.dimensions(first_pass.numbers.widths.hidden.clone()),
            ]
            .concat(),
        )?;
        let tensor = rows.tensor().clone();
        (DraftConditioning::Rows(rows), tensor)
    };
    let mut injection = Vec::with_capacity(layer_count);
    let mut injected = None;
    for (index, (layer_weight, layer)) in layer_weights.iter().zip(&entries.layers).enumerate() {
        let (Some(weights_of), Some(pass)) = (layer_weight, passes.injection(index)) else {
            injection.push(None);
            continue;
        };
        let (mixed, state, controls) = attention_graph::attention(
            &mut graph,
            layer.injection,
            &pass.topology,
            &pass.numbers,
            weights_of,
            &mut constants,
            SublayerInput::single(&conditioning_rows),
            &conditioning_rows,
            &Join::single(Some(ENTRY_ROWS)),
        )?;
        injection.push(Some(LayerPorts { controls, planes: state.planes }));
        injected = Some(mixed);
    }
    let injected = injected.ok_or("a draft has no layers")?;
    graph.set_class_scope(None);
    let ClassPasses::Drafting(drafting) = passes else {
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
    };

    // The block pass. Its rows, proposals and selections are counts of the
    // class and the draft's block (its topology's), its widths the model's.
    // Its unscaled weights bind the graph's absent operand
    // (`GraphConstant::absent`).
    let (slots, proposals, block_rows) = (class.slots, class.proposals, class.block_rows);
    let rows = slots * block_rows;
    let corrector = match (&numbers.method, &entries.corrector) {
        (MethodNumbers::LiLiCorr { topology, widths }, Some(stages)) => Some((topology, widths, stages)),
        (MethodNumbers::LiLiCorr { .. }, None) | (_, Some(_)) => {
            return Err("draft method and corrector entries disagree".into());
        }
        _ => None,
    };
    // The selections, step-major; a corrector projects every mask row of
    // every slot (slot-major), the rest each proposing row.
    let selected = proposals * slots;
    let mask_rows = model.mask_rows;
    let outputs = match corrector {
        Some(_) => slots * mask_rows,
        None => selected,
    };
    // The block's tokens are the host's (`BoundDraftGraphs::embedding`).
    let mut table = GatheredTable::of(&mut graph, planned, model.token_scope, WeightKind::Embedding, &mut weights)?;
    // The empty directory: every table this graph gathers, it gathers by
    // the host's rows or the rows of a table held whole (`gathered_table`).
    let direct = table.by_rows(&mut graph, &mut constants)?;
    let embedding_dims = [
        ("M", Dim::from(rows)),
        ("V", table.rows.clone()),
        ("D", hidden.clone()),
        ("R", direct.shape[0].clone()),
        ("F", direct.shape[1].clone()),
    ];
    let tokens = graph.input_for(entries.embedding, "tokens", &embedding_dims)?;
    let mut residual = graph
        .enqueue(
            entries.embedding,
            &embedding_dims,
            embedding_rows::WorkflowArgs {
                table: (&table.table).into(),
                tokens: tokens.tensor().into(),
                directory: (&direct.tensor).into(),
                scale: embedding_scale,
                normalize: 0.into(),
                epsilon,
            },
        )?
        .r1;
    let mut layers = Vec::with_capacity(layer_count);
    // A convolved backbone's unfused layers read their normed rows through
    // the identity row map of the block rows.
    let convolved = match (&numbers.convolution, &entries.convolved) {
        (Some([kernel, group]), Some(stages)) => {
            let identity = GraphConstant::row_identity(&mut graph, rows, None)?;
            let rows_map = identity.port().tensor().clone();
            constants.push(identity);
            Some((stages, kernel, group, rows_map))
        }
        (None, None) => None,
        _ => return Err("draft convolutions and entries disagree".into()),
    };
    for (index, ((layer_weight, layer), drafting)) in
        layer_weights.iter().zip(&entries.layers).zip(drafting).enumerate()
    {
        // A drafting class injects every layer.
        let injection_weights = layer_weight.as_ref().expect("a drafting class injects every layer");
        let (topology, block) = (&drafting.block.topology, &drafting.block.numbers);
        if let Some((stages, kernel, group, rows_map)) = &convolved {
            let ports;
            (residual, ports) = convolved_layer(
                &mut graph,
                ConvolvedLayer {
                    entries: &stages.layers[index],
                    convolve_input: stages.convolve_input,
                    convolve_residual: stages.convolve_residual,
                    gated: stages.gated,
                    mix: layer.attention.history,
                    topology,
                    block,
                    epsilon,
                    index,
                    attention_weights: injection_weights,
                    rows_map,
                    block_rows,
                    kernel,
                    group,
                    hidden,
                    intermediate: &numbers.dense[index].features,
                },
                planned,
                &mut weights,
                &mut constants,
                &residual,
            )?;
            layers.push(ports);
            continue;
        }
        let input_norm = weight(&mut graph, planned, draft_sublayer(index, 0)?, WeightKind::InputNorm, &mut weights)?;
        let block_weights = AttentionWeights {
            input_norm: Some(input_norm),
            query: injection_weights.query.clone(),
            low_rank_query: injection_weights.low_rank_query.clone(),
            gate: injection_weights.gate.clone(),
            key: injection_weights.key.clone(),
            value: injection_weights.value.clone(),
            query_norm: injection_weights.query_norm.clone(),
            key_norm: injection_weights.key_norm.clone(),
            sinks: injection_weights.sinks.clone(),
            output: injection_weights.output.clone(),
            grouped: injection_weights.grouped.clone(),
            post_norm: None,
            operands: injection_weights.operands.clone(),
            indexer: None,
            picked: None,
        };
        let (mixed, state, controls) = attention_graph::attention(
            &mut graph,
            layer.attention,
            topology,
            block,
            &block_weights,
            &mut constants,
            SublayerInput::single(&residual),
            &residual,
            &Join::single(None),
        )?;
        layers.push(LayerPorts { controls, planes: state.planes });
        residual = dense_graph::dense(
            &mut graph,
            DenseGraphEntries { expand: layer.dense.expand, output: layer.dense.output },
            model.expansions[index],
            &numbers.dense[index],
            planned,
            draft_sublayer(index, 1)?,
            &mut weights,
            &mut constants,
            &mixed,
            &Dim::from(rows),
            None,
            epsilon,
            None,
        )?;
    }
    let output_norm = weight(&mut graph, planned, WeightScope::Draft, WeightKind::OutputNorm, &mut weights)?;
    let readout_vocabulary = vocabulary.clone();
    let mut leading = Vec::new();
    // Full draft logits read the projection's vocabulary prefix as it is
    // stored: the packed projection, or the planes of a scope another reader
    // selected a class of (`ModelLoadPlan::finalize_readout`), through their
    // full pass.
    let projection = model.projection.ports(
        &mut graph,
        false,
        &mut VocabularyPrefix {
            planned,
            vocabulary: &readout_vocabulary,
            leading: &mut leading,
            weights: &mut weights,
            constants: &mut constants,
        },
    )?;
    let mut dimensions = vec![
        ("M", Dim::from(rows)),
        ("O", Dim::from(outputs)),
        ("V", readout_vocabulary.clone()),
        ("D", hidden.clone()),
        ("PT", Dim::from(1)),
        ("NM", hidden.clone()),
    ];
    match &projection {
        readout::ProjectionPorts::Packed { scale_extent, .. } => dimensions.push(("WS", scale_extent.clone())),
        readout::ProjectionPorts::Progressive { depth, .. } => {
            dimensions.push(("P", Dim::from(u64::from(depth.rest_planes()))));
        }
    }
    let head_rows = entries.head.raw_out_rows(&mut graph, &projection, &dimensions)?;
    let mut logits = readout::project_readout(
        &mut graph,
        entries.head.borrowed(),
        &projection,
        readout::ReadoutInput::RawRms {
            hidden: &residual,
            norm: &output_norm,
            form: InputForm::SINGLE,
            out_rows: head_rows.tensor(),
            epsilon,
        },
        readout::ProjectionGeometry {
            rows: Dim::from(rows),
            projected: Dim::from(outputs),
            selected: Dim::from(0),
            vocabulary: readout_vocabulary.clone(),
            hidden: hidden.clone(),
            columns: readout_vocabulary.clone(),
        },
        softcap,
        Some(entries.sample),
        &mut constants,
    )?
    .logits;
    let result = graph.local_for(entries.sample, "result", &[("M", Dim::from(selected)), ("V", readout_vocabulary.clone())])?;
    // Each proposal's probability under the logits it was sampled from,
    // step-major beside the selections, of a method that samples them.
    let confidence = model
        .samples
        .then(|| {
            Ok::<_, GraphError>(
                graph
                    .local_for(
                        entries.confidence,
                        "confidence",
                        &[("M", Dim::from(selected)), ("N", Dim::from(outputs)), ("V", readout_vocabulary.clone())],
                    )?
                    .tensor()
                    .clone(),
            )
        })
        .transpose()?;
    let sampled = || {
        confidence
            .as_ref()
            .ok_or_else(|| GraphError::from("a sampling draft has no confidences".to_owned()))
    };
    let selection = (entries.shape, entries.sample, entries.confidence);
    let selection_entries = move || readout::SelectionEntries {
        shape: selection.0,
        sample: selection.1,
        confidence: selection.2,
    };
    let mut selections = Vec::new();
    let mut anchor_rows = None;
    let mut landed = None;
    let anchors = match (&numbers.method, &entries.markov) {
        (MethodNumbers::DFlash, None) => {
            let mut all = result.tensor().slice_leading(0, outputs);
            let mut all_confidence = sampled()?.slice_leading(0, outputs);
            let identity = GraphConstant::identity(&mut graph, outputs)?;
            let inputs =
                readout::selection_inputs(&mut graph, entries.sample, &Dim::from(outputs), &readout_vocabulary)?;
            selections.push(readout::select(
                &mut graph,
                selection_entries(),
                &readout_vocabulary,
                &mut logits,
                None,
                identity.port().tensor(),
                &Dim::from(outputs),
                class.shaped,
                inputs,
                &mut all,
                &mut all_confidence,
            )?);
            constants.push(identity);
            None
        }
        (MethodNumbers::DSpark { rank }, Some(chain)) => {
            let markov_table =
                weight(&mut graph, planned, WeightScope::Draft, WeightKind::MarkovEmbedding, &mut weights)?;
            let markov_role = WeightRole { scope: WeightScope::Draft, kind: WeightKind::MarkovProjection };
            // Its bias adds one logit per readout vocabulary row.
            let markov_projection =
                leading_port(&mut graph, planned, markov_role, &readout_vocabulary, rank, &mut leading)?;
            let memory_dims = [
                ("M", Dim::from(slots)),
                ("V", vocabulary.clone()),
                ("D", rank.clone()),
                ("R", direct.shape[0].clone()),
                ("F", direct.shape[1].clone()),
            ];
            let anchors = graph.input_for(chain.embedding, "tokens", &memory_dims)?;
            // The projection's packing bound (a bound of all its rows bounds
            // the readout vocabulary's).
            let markov_bound = PackingBound::planned(&mut graph, planned, markov_role, &mut weights, &mut constants)?;
            // An NVFP4 projection's second-level scale (absent otherwise).
            let (markov_scale, markov_scale_extent) =
                weight_scale_port(&mut graph, planned, markov_role, &mut weights, &mut constants)?;
            // A step selects from its own biased rows, row `s` slot `s`.
            let identity = GraphConstant::identity(&mut graph, slots)?;
            for step in 0..proposals {
                let rows_of_step = GraphConstant::i32(
                    &mut graph,
                    &(step * slots..(step + 1) * slots)
                        .map(|row| i32::try_from(row).map_err(|_| "draft output row exceeds i32"))
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                let previous = match step {
                    0 => anchors.tensor().slice_leading(0, slots),
                    _ => result.tensor().slice_leading((step - 1) * slots, step * slots),
                };
                let memory = graph
                    .enqueue(
                        chain.embedding,
                        &memory_dims,
                        embedding_rows::WorkflowArgs {
                            table: (&markov_table).into(),
                            tokens: (&previous).into(),
                            directory: (&direct.tensor).into(),
                            scale: 1.0.into(),
                            normalize: 0.into(),
                            epsilon,
                        },
                    )?
                    .r0;
                let mut biased = graph
                    .enqueue(
                        chain.projection,
                        &[
                            ("M", Dim::from(outputs)),
                            ("O", Dim::from(slots)),
                            ("H", readout_vocabulary.clone()),
                            ("F", rank.clone()),
                            ("DS", markov_scale_extent.clone()),
                            ("DB", markov_bound.extent.clone()),
                        ],
                        dense_output::WorkflowArgs {
                            residual: (&logits).into(),
                            product: (&memory).into(),
                            down_weight: (&markov_projection).into(),
                            out_rows: rows_of_step.port().tensor().into(),
                            down_scale: markov_scale.tensor().into(),
                            down_bound: (&markov_bound.tensor).into(),
                        },
                    )?
                    .value;
                let mut step_result = result.tensor().slice_leading(step * slots, (step + 1) * slots);
                let inputs =
                    readout::selection_inputs(&mut graph, entries.sample, &Dim::from(slots), &readout_vocabulary)?;
                let mut step_confidence = sampled()?.slice_leading(step * slots, (step + 1) * slots);
                // Its proposals' confidence is their probability under the
                // biased logits they are drawn from. DSpark's own confidence
                // head (sigmoid(score) < 0 at the service's fixed threshold)
                // cannot decline a selection (including a NaN score), so
                // neither its score nor its features are computed.
                selections.push(readout::select(
                    &mut graph,
                    selection_entries(),
                    &readout_vocabulary,
                    &mut biased,
                    None,
                    identity.port().tensor(),
                    &Dim::from(slots),
                    class.shaped,
                    inputs,
                    &mut step_result,
                    &mut step_confidence,
                )?);
                constants.push(rows_of_step);
            }
            constants.push(identity);
            Some(anchors)
        }
        (MethodNumbers::DFlash2 { top_k, rank }, None) => {
            let stages = entries.selector.as_ref().ok_or("a DFlash2 draft graph has no selector entries")?;
            Some(selector_path(
                &mut graph,
                stages,
                planned,
                &mut weights,
                &mut constants,
                SelectorPath {
                    direct: &direct,
                    residual: &residual,
                    output_norm: &output_norm,
                    head_rows: head_rows.tensor(),
                    logits: &logits,
                    result: result.tensor(),
                    rows,
                    slots,
                    proposals,
                    vocabulary,
                    readout_vocabulary: &readout_vocabulary,
                    hidden,
                    rank,
                    top_k,
                    epsilon,
                },
            )?)
        }
        (MethodNumbers::LiLiCorr { .. }, None) => {
            let (corrector, widths, stages) = corrector.ok_or("a LiLiCorr draft graph has no corrector entries")?;
            let (rows, read) = correct_candidates(
                &mut graph,
                stages,
                entries.embedding,
                entries.landing,
                planned,
                &mut weights,
                &mut constants,
                CorrectorPath {
                    topology: reals.arguments(corrector.clone()),
                    widths,
                    table: &mut table,
                    conditioning: &conditioning_rows,
                    entry_rows,
                    fusion_norm: &fusion_norm,
                    residual: &residual,
                    output_norm: &output_norm,
                    head_rows: head_rows.tensor(),
                    logits: &logits,
                    result: result.tensor(),
                    rows,
                    slots,
                    mask_rows,
                    proposals,
                    readout_vocabulary: &readout_vocabulary,
                    hidden,
                    embedding_scale,
                    epsilon,
                },
            )?;
            anchor_rows = Some(rows);
            landed = read;
            None
        }
        _ => return Err("draft method and entries disagree".into()),
    };
    // A selector's or corrector's proposals are chosen by path scores over
    // a few candidates, which give no probability: they carry none.
    if let Some(confidence) = &confidence {
        graph.export(confidence)?;
    }
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
            anchor_rows,
            landed,
            leading,
            readout_vocabulary,
            confidence,
        }),
        constants,
        weights,
        output,
    })
}

/// One `project_rows` of `rows` rows through `weight`, binding its planned
/// scale and packing bound (`operands`).
fn project_planned<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Binding<'a, project_rows::Entry>,
    (weight, operands): (&WorkflowTensor, &WeightOperands),
    source: seismic::WorkflowTensorRef<'_>,
    (rows, inputs, outputs): (Dim, Dim, Dim),
) -> Result<WorkflowTensor, GraphError> {
    graph
        .enqueue(
            entry,
            &[
                ("M", rows),
                ("G", Dim::from(1)),
                ("K", inputs),
                ("N", outputs),
                ("WS", operands.scale.extent.clone()),
                ("WB", operands.bound.extent.clone()),
            ],
            project_rows::WorkflowArgs {
                source,
                weight: weight.into(),
                weight_scale: (&operands.scale.tensor).into(),
                weight_bound: (&operands.bound.tensor).into(),
            },
        )
        .map(|projected| projected.value)
}

/// [`project_planned`] of the planned weight of `scope` and `kind`, loading
/// its ports.
#[allow(clippy::too_many_arguments)]
fn project_role<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Binding<'a, project_rows::Entry>,
    planned: &GraphWeights,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    (scope, kind): (WeightScope, WeightKind),
    source: seismic::WorkflowTensorRef<'_>,
    dims: (Dim, Dim, Dim),
) -> Result<WorkflowTensor, GraphError> {
    let weight_tensor = weight(graph, planned, scope, kind, weights)?;
    let operands = WeightOperands::planned(graph, planned, WeightRole { scope, kind }, weights, constants)?;
    project_planned(graph, entry, (&weight_tensor, &operands), source, dims)
}

/// One convolved draft layer of the block pass and what it reads.
struct ConvolvedLayer<'a, 'o, G: GraphDraft + 'a> {
    entries: &'o ConvolvedLayerEntries<'a, G>,
    convolve_input: G::Binding<'a, draft_convolve_input::Entry>,
    convolve_residual: G::Binding<'a, draft_convolve_residual::Entry>,
    gated: G::Binding<'a, draft_gated_rows::Entry>,
    /// The layer's attention entry of its history codec.
    mix: AttentionHistoryEntries<'a, G>,
    /// The layer's attention in the block pass: its topology and its
    /// numbers (its rows the block rows).
    topology: &'o AttentionTopology,
    block: &'o AttentionBlock,
    /// Its norms' epsilon.
    epsilon: Argument<f32>,
    index: usize,
    /// The layer's projection and head-norm weight ports.
    attention_weights: &'o AttentionWeights,
    /// The identity row map of the block rows.
    rows_map: &'o WorkflowTensor,
    /// Rows of each slot's block (`DraftDefinition::block_rows`).
    block_rows: u64,
    kernel: &'o Dim,
    group: &'o Dim,
    hidden: &'o Dim,
    /// The feed-forward's expansion width.
    intermediate: &'o Dim,
}

/// One convolved layer over the block rows (upstream
/// `Qwen3DFlashDecoderLayer` with its convolutions): for the attention and
/// then the feed-forward, the normed rows project to their convolution
/// coefficients, the half-0 convolution feeds the operator's plain
/// projections, and the half-1 convolution of the operator's F32 output
/// joins the residual. The attention appends nothing (its destinations are
/// the block pass's −1) and reads the injected context and the whole fresh
/// block.
#[allow(clippy::too_many_arguments)]
fn convolved_layer<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    layer: ConvolvedLayer<'a, '_, G>,
    planned: &GraphWeights,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
) -> Result<(WorkflowTensor, LayerPorts), GraphError> {
    let ConvolvedLayer { entries, hidden, kernel, group, .. } = layer;
    let rows = &layer.block.rows;
    let epsilon = layer.epsilon;
    let groups = hidden / group;
    let coefficients = kernel * 2 * &groups;
    let convolution_dims = [("M", rows.clone()), ("G", groups.clone()), ("C", group.clone()), ("K", kernel.clone())];
    let block = u32::try_from(layer.block_rows).map_err(|_| "draft block exceeds u32")?;
    // A projection of the block rows binds its weight's planned scale and
    // packing bound: the attention's through its loaded weights and
    // operands, the rest loading theirs.
    let project_attention = |graph: &mut G,
                             entry: G::Binding<'a, project_rows::Entry>,
                             operand: (&WorkflowTensor, &WeightOperands),
                             source: seismic::WorkflowTensorRef<'_>,
                             (inputs, outputs): (Dim, Dim)| {
        project_planned(graph, entry, operand, source, (rows.clone(), inputs, outputs))
    };
    let project = |graph: &mut G,
                   weights: &mut Vec<(WeightPort, NativePort)>,
                   constants: &mut Vec<GraphConstant>,
                   entry: G::Binding<'a, project_rows::Entry>,
                   role: (WeightScope, WeightKind),
                   source: seismic::WorkflowTensorRef<'_>,
                   (inputs, outputs): (Dim, Dim)| {
        project_role(graph, entry, planned, weights, constants, role, source, (rows.clone(), inputs, outputs))
    };
    // One sublayer's prologue: its normed rows, their coefficients and the
    // half-0 convolution the operator reads.
    let prologue = |graph: &mut G,
                    weights: &mut Vec<(WeightPort, NativePort)>,
                    constants: &mut Vec<GraphConstant>,
                    residual: &WorkflowTensor,
                    sublayer: u32,
                    norm_entry: G::Binding<'a, readout_features_rows::Entry>,
                    coefficient_entry: G::Binding<'a, project_rows::Entry>| {
        let scope = draft_sublayer(layer.index, sublayer)?;
        let norm = weight(graph, planned, scope, WeightKind::InputNorm, weights)?;
        let base = weight(graph, planned, scope, WeightKind::ConvolutionBase, weights)?;
        let normed = graph
            .enqueue(
                norm_entry,
                &[
                    ("M", rows.clone()),
                    ("O", rows.clone()),
                    ("D", hidden.clone()),
                    ("PT", Dim::from(1)),
                    ("NM", hidden.clone()),
                ],
                readout_features_rows::WorkflowArgs {
                    hidden: (&residual.reshape(&[rows, &Dim::from(1), hidden])).into(),
                    norm: (&norm).into(),
                    out_rows: layer.rows_map.into(),
                    epsilon,
                },
            )?
            .value;
        let dynamic = project(
            graph,
            weights,
            constants,
            coefficient_entry,
            (scope, WeightKind::ConvolutionProjection),
            (&normed).into(),
            (hidden.clone(), coefficients.clone()),
        )?;
        let dynamic = dynamic.reshape(&[rows, &Dim::from(2), kernel, &groups]);
        let convolved = graph
            .enqueue(
                layer.convolve_input,
                &convolution_dims,
                draft_convolve_input::WorkflowArgs {
                    input: (&normed).into(),
                    dynamic: (&dynamic).into(),
                    base: (&base).into(),
                    block: block.into(),
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
                        block: block.into(),
                    },
                )
                .map(|finished| finished.value)
        };

    // Attention: plain query, key and value projections of the convolved
    // rows, the history codec's attention, and the output projection.
    let (convolved, dynamic, base) = prologue(
        graph,
        weights,
        constants,
        residual,
        0,
        entries.attention_norm,
        entries.attention_coefficients,
    )?;
    let numbers = layer.block;
    let (heads, width) = (numbers.widths.heads(), &numbers.widths.width);
    let key_width = numbers.widths.key_width();
    let attention = layer.attention_weights;
    let (Some(key_weight), Some(value_weight)) = (&attention.key, &attention.value) else {
        return Err("a convolved draft layer projects its keys and values".into());
    };
    if attention.low_rank_query.is_some() {
        return Err("a convolved draft layer projects its query whole".into());
    }
    let query = project_attention(
        graph,
        entries.query,
        (&attention.query, &attention.operands.query),
        (&convolved).into(),
        (hidden.clone(), &heads * width),
    )?;
    let key = project_attention(
        graph,
        entries.key,
        (key_weight, &attention.operands.key),
        (&convolved).into(),
        (hidden.clone(), key_width.clone()),
    )?;
    let value = project_attention(
        graph,
        entries.value,
        (value_weight, &attention.operands.value),
        (&convolved).into(),
        (hidden.clone(), key_width.clone()),
    )?;
    let one = Dim::from(1);
    let (attended, state, controls) = attention_graph::mix(
        graph,
        layer.mix,
        (&attention.query_norm, &attention.key_norm),
        &attention.sinks,
        constants,
        &ProjectedRows {
            query: query.reshape(&[rows, &heads, width]),
            // No gate: zero columns of every row.
            gate: query.slice_leading(0, 0).reshape(&[rows, &heads, &Dim::from(0)]),
            key: key.reshape(&[&one, rows, &key_width]),
            value: value.reshape(&[&one, rows, &key_width]),
        },
        layer.topology,
        numbers,
        None,
        None,
        None,
    )?;
    let attended = attended.ok_or("draft block attention result is absent")?;
    let output = project_attention(
        graph,
        entries.output,
        (
            attention.output.as_ref().ok_or("draft block output weight is absent")?,
            &attention.operands.output,
        ),
        (&attended.reshape(&[rows, &(&heads * width)])).into(),
        (&heads * width, hidden.clone()),
    )?;
    let residual = finish(graph, residual, &output, (&dynamic, &base))?;

    // Feed-forward: gate and up projections of the convolved rows, their
    // gated product and the down projection.
    let (convolved, dynamic, base) = prologue(
        graph,
        weights,
        constants,
        &residual,
        1,
        entries.feed_forward_norm,
        entries.feed_forward_coefficients,
    )?;
    let scope = draft_sublayer(layer.index, 1)?;
    let features = layer.intermediate;
    let gate = project(
        graph,
        weights,
        constants,
        entries.gate,
        (scope, WeightKind::DenseGate),
        (&convolved).into(),
        (hidden.clone(), features.clone()),
    )?;
    let up = project(
        graph,
        weights,
        constants,
        entries.up,
        (scope, WeightKind::DenseUp),
        (&convolved).into(),
        (hidden.clone(), features.clone()),
    )?;
    let product = graph
        .enqueue(
            layer.gated,
            &[("M", rows.clone()), ("F", features.clone())],
            draft_gated_rows::WorkflowArgs { gate: (&gate).into(), up: (&up).into() },
        )?
        .value;
    let down = project(
        graph,
        weights,
        constants,
        entries.down,
        (scope, WeightKind::DenseDown),
        (&product).into(),
        (features.clone(), hidden.clone()),
    )?;
    let residual = finish(graph, &residual, &down, (&dynamic, &base))?;
    Ok((residual, LayerPorts { controls, planes: state.planes }))
}

/// The block pass's rows DFlash2's candidate path reads.
struct SelectorPath<'t> {
    /// The empty directory the codebooks are read through.
    direct: &'t RowDirectory,
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
    vocabulary: &'t Dim,
    readout_vocabulary: &'t Dim,
    hidden: &'t Dim,
    rank: &'t Dim,
    top_k: &'t Dim,
    epsilon: Argument<f32>,
}

/// DFlash2's ordered candidate path (upstream `CandidateSelector.select`,
/// greedy): every proposing row's top-k candidates and their logits, its
/// output-normed row's selector projection and its candidates' successor
/// codes; then per step, the predecessor code of the token each slot
/// follows (its anchor, then its previous selection) and the step's
/// selection. Returns the anchors' input port.
fn selector_path<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: &SelectorEntries<'a, G>,
    planned: &GraphWeights,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    path: SelectorPath<'_>,
) -> Result<NativePort, GraphError> {
    let SelectorPath {
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
        planned,
        WeightScope::Draft,
        WeightKind::SelectorHidden,
        weights,
    )?;
    let predecessor_codes = weight(
        graph,
        planned,
        WeightScope::Draft,
        WeightKind::SelectorPredecessor,
        weights,
    )?;
    let successor_codes = weight(
        graph,
        planned,
        WeightScope::Draft,
        WeightKind::SelectorSuccessor,
        weights,
    )?;
    let top_k_dims = [("M", Dim::from(outputs)), ("V", readout_vocabulary.clone()), ("K", top_k.clone())];
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
            &[
                ("M", Dim::from(path.rows)),
                ("O", Dim::from(outputs)),
                ("D", hidden.clone()),
                ("PT", Dim::from(1)),
                ("NM", hidden.clone()),
            ],
            readout_features_rows::WorkflowArgs {
                hidden: (&path.residual.reshape(&[&Dim::from(path.rows), &Dim::from(1), hidden])).into(),
                norm: path.output_norm.into(),
                out_rows: path.head_rows.into(),
                epsilon: path.epsilon,
            },
        )?
        .value;
    let hidden_operands = WeightOperands::planned(
        graph,
        planned,
        WeightRole {
            scope: WeightScope::Draft,
            kind: WeightKind::SelectorHidden,
        },
        weights,
        constants,
    )?;
    let projected = project_planned(
        graph,
        entries.hidden,
        (&selector_hidden, &hidden_operands),
        (&features).into(),
        (Dim::from(outputs), hidden.clone(), rank.clone()),
    )?;
    let successors = graph
        .enqueue(
            entries.successor,
            &[
                ("M", top_k * outputs),
                ("V", vocabulary.clone()),
                ("D", rank.clone()),
                ("R", path.direct.shape[0].clone()),
                ("F", path.direct.shape[1].clone()),
            ],
            embedding_rows::WorkflowArgs {
                table: (&successor_codes).into(),
                tokens: (&candidates).into(),
                directory: (&path.direct.tensor).into(),
                scale: 1.0.into(),
                normalize: 0.into(),
                epsilon: path.epsilon,
            },
        )?
        .r0;
    let code_dims = [
        ("M", Dim::from(slots)),
        ("V", vocabulary.clone()),
        ("D", rank.clone()),
        ("R", path.direct.shape[0].clone()),
        ("F", path.direct.shape[1].clone()),
    ];
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
                    directory: (&path.direct.tensor).into(),
                    scale: 1.0.into(),
                    normalize: 0.into(),
                    epsilon: path.epsilon,
                },
            )?
            .r0;
        let mut selection = path.result.slice_leading(first, last);
        graph.enqueue(
            entries.path,
            &[("S", Dim::from(slots)), ("K", top_k.clone()), ("R", rank.clone())],
            draft_path_step::WorkflowArgs {
                candidates: (&candidates.slice_leading(top_k * first, top_k * last)).into(),
                unary: (&unary.slice_leading(first, last)).into(),
                hidden: (&projected.slice_leading(first, last)).into(),
                predecessor: (&predecessors).into(),
                successor: (&successors.slice_leading(top_k * first, top_k * last)).into(),
                selection: (&mut selection).into(),
            },
        )?;
    }
    Ok(anchors)
}

/// The block pass's rows and tables LiLiCorr's corrector reads.
struct CorrectorPath<'t> {
    topology: CorrectorTopology<Argument<f64>>,
    widths: &'t CorrectorNumbers<Dim>,
    /// The block's token table, which the candidates' embeddings gather.
    table: &'t mut GatheredTable,
    /// The entry rows' F32 conditioning `[entry_rows, hidden]`.
    conditioning: &'t WorkflowTensor,
    entry_rows: &'t Dim,
    fusion_norm: &'t WorkflowTensor,
    residual: &'t WorkflowTensor,
    output_norm: &'t WorkflowTensor,
    /// The block row of every mask row, slot-major.
    head_rows: &'t WorkflowTensor,
    /// The mask rows' readout logits `[slots · mask_rows, readout_vocabulary]`.
    logits: &'t WorkflowTensor,
    /// The selections `[proposals · slots, 2]`, step-major.
    result: &'t WorkflowTensor,
    rows: u64,
    slots: u64,
    mask_rows: u64,
    proposals: u64,
    readout_vocabulary: &'t Dim,
    hidden: &'t Dim,
    embedding_scale: Argument<f32>,
    epsilon: Argument<f32>,
}

/// The weight kind of a corrector weight.
fn corrector_weight(weight: CorrectorWeight) -> WeightKind {
    WeightKind::Corrector(weight)
}

/// The weight kind of part `part` of the corrector's affine map at `site`.
fn corrector_affine(site: CorrectorSite, part: AffinePart) -> WeightKind {
    corrector_weight(CorrectorWeight::Affine { site, part })
}

/// LiLiCorr's candidate correction (upstream `build_lilicorr_scores` and
/// its greedy walk): every mask row's top-k candidates; their nodes (the
/// candidate's token embedding, the mask row's output-normed pass row, the
/// candidate's softmax features, and slot and rank embeddings); the
/// corrector's layers over each slot's nodes; its output norm; each slot's
/// anchor state from the fused feature that selected its anchor; the
/// factored outgoing, incoming and anchor vectors; and the walk selecting
/// each slot's first `proposals` tokens. Returns the anchor rows' input port
/// and, of a token table with landing slots, the landed read of the
/// candidates' embeddings (rows the device selects).
#[allow(clippy::too_many_arguments)]
fn correct_candidates<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: &CorrectorEntries<'a, G>,
    embedding: G::Binding<'a, embedding_rows::Entry>,
    landing: Option<LandingEntries<'a, G>>,
    planned: &GraphWeights,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    path: CorrectorPath<'_>,
) -> Result<(NativePort, Option<LandedReadPorts>), GraphError> {
    let (slots, mask_rows, hidden, epsilon) = (path.slots, path.mask_rows, path.hidden, path.epsilon);
    let (width, heads, factor, mlp) = (&path.widths.width, &path.widths.heads, &path.widths.factor, &path.widths.mlp);
    let top_k = path.topology.top_k;
    let scalars = &path.topology;
    if entries.layers.len() != scalars.layers.len() {
        return Err("corrector entries disagree with its layers".into());
    }
    let head_width = width / heads;
    let outputs = slots * mask_rows;
    // Each block's nodes, and every node of every block.
    let nodes = mask_rows * top_k;
    let node_rows = slots * nodes;
    let draft = WeightScope::Draft;
    let (one, three) = (Dim::from(1), Dim::from(3));
    let load_weight = |graph: &mut G, weights: &mut Vec<(WeightPort, NativePort)>, scope: WeightScope, kind: WeightKind| {
        weight(graph, planned, scope, kind, weights)
    };
    let project = |graph: &mut G,
                   weights: &mut Vec<(WeightPort, NativePort)>,
                   constants: &mut Vec<GraphConstant>,
                   entry: G::Binding<'a, project_rows::Entry>,
                   (scope, site): (WeightScope, CorrectorSite),
                   source: seismic::WorkflowTensorRef<'_>,
                   dims: (Dim, Dim, Dim)| {
        project_role(
            graph,
            entry,
            planned,
            weights,
            constants,
            (scope, corrector_affine(site, AffinePart::Weight)),
            source,
            dims,
        )
    };
    // One F32 row op after a projection (`corrector_rows`): its map's bias,
    // then the form's activation and the residual rows, if any.
    let finish = |graph: &mut G,
                  weights: &mut Vec<(WeightPort, NativePort)>,
                  constants: &mut Vec<GraphConstant>,
                  entry: G::Binding<'a, corrector_rows::Entry>,
                  (scope, site): (WeightScope, CorrectorSite),
                  values: &WorkflowTensor,
                  residual: Option<&WorkflowTensor>,
                  (form, epsilon): (i32, Argument<f32>),
                  (rows, columns): (Dim, Dim)| {
        let bias = weight(graph, planned, scope, corrector_affine(site, AffinePart::Bias), weights)?;
        let (residual, residual_rows) = match residual {
            Some(residual) => (residual.reshape(&[&Dim::from(1), &rows, &columns]), 1),
            None => (
                GraphConstant::absent(graph, constants)?.tensor().reshape(&[&Dim::from(0), &rows, &columns]),
                0,
            ),
        };
        graph
            .enqueue(
                entry,
                &[("M", rows), ("N", columns), ("R", Dim::from(residual_rows))],
                corrector_rows::WorkflowArgs {
                    values: values.into(),
                    bias: (&bias).into(),
                    residual: (&residual).into(),
                    form: form.into(),
                    epsilon,
                },
            )
            .map(|finished| finished.value)
    };
    // An RMS norm of every row of `rows` (`[count, columns]` F32).
    let norm = |graph: &mut G,
                constants: &mut Vec<GraphConstant>,
                entry: G::Binding<'a, readout_features_rows::Entry>,
                rows: &WorkflowTensor,
                weight: &WorkflowTensor,
                (count, columns): (u64, &Dim),
                epsilon: Argument<f32>| {
        let identity = GraphConstant::identity(graph, count)?;
        let out_rows = identity.port().tensor().clone();
        constants.push(identity);
        let count = Dim::from(count);
        graph
            .enqueue(
                entry,
                &[
                    ("M", count.clone()),
                    ("O", count.clone()),
                    ("D", columns.clone()),
                    ("PT", Dim::from(1)),
                    ("NM", columns.clone()),
                ],
                readout_features_rows::WorkflowArgs {
                    hidden: (&rows.reshape(&[&count, &Dim::from(1), columns])).into(),
                    norm: weight.into(),
                    out_rows: (&out_rows).into(),
                    epsilon,
                },
            )
            .map(|normed| normed.value)
    };

    // Every mask row's candidates, in descending logit order.
    let top_k_dims = [
        ("M", Dim::from(outputs)),
        ("V", path.readout_vocabulary.clone()),
        ("K", Dim::from(top_k)),
    ];
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

    // The candidates' feature rows.
    let feature_norm = (
        load_weight(graph, weights, draft, corrector_weight(CorrectorWeight::FeatureNorm(AffinePart::Weight)))?,
        load_weight(graph, weights, draft, corrector_weight(CorrectorWeight::FeatureNorm(AffinePart::Bias)))?,
    );
    let feature_up = (
        load_weight(graph, weights, draft, corrector_affine(CorrectorSite::FeatureUp, AffinePart::Weight))?,
        load_weight(graph, weights, draft, corrector_affine(CorrectorSite::FeatureUp, AffinePart::Bias))?,
    );
    let features = graph
        .enqueue(
            entries.features,
            &[
                ("M", Dim::from(outputs)),
                ("V", path.readout_vocabulary.clone()),
                ("K", Dim::from(top_k)),
                ("H", width.clone()),
            ],
            corrector_features::WorkflowArgs {
                logits: path.logits.into(),
                unary: (&unary).into(),
                norm_weight: (&feature_norm.0).into(),
                norm_bias: (&feature_norm.1).into(),
                up_weight: (&feature_up.0).into(),
                up_bias: (&feature_up.1).into(),
                epsilon: scalars.feature.f32(),
            },
        )?
        .value;
    let feature = project(
        graph,
        weights,
        constants,
        entries.feature_down,
        (draft, CorrectorSite::FeatureDown),
        (&features).into(),
        (Dim::from(node_rows), width.clone(), width.clone()),
    )?;

    // The candidates' token rows (rows the device selects, a landed read of
    // a table with landing slots): their embeddings, projected into the
    // corrector width unless it is the draft's.
    let (gathered, table_rows) = (path.table.table.clone(), path.table.rows.clone());
    let (embedded, landed) = path.table.read_selected(
        graph,
        planned,
        weights,
        constants,
        landing,
        (&candidates).into(),
        &Dim::from(node_rows),
        None,
        |graph, directory| {
            Ok(graph.enqueue(
                embedding,
                &[
                    ("M", Dim::from(node_rows)),
                    ("V", table_rows.clone()),
                    ("D", hidden.clone()),
                    ("R", directory.shape[0].clone()),
                    ("F", directory.shape[1].clone()),
                ],
                embedding_rows::WorkflowArgs {
                    table: (&gathered).into(),
                    tokens: (&candidates).into(),
                    directory: (&directory.tensor).into(),
                    scale: path.embedding_scale,
                    normalize: 0.into(),
                    epsilon,
                },
            )?)
        },
    )?;
    let (token, token_bias, token_biases) = match &entries.token {
        Some(entry) => {
            let token = project(
                graph,
                weights,
                constants,
                *entry,
                (draft, CorrectorSite::Token),
                (&embedded.r0).into(),
                (Dim::from(node_rows), hidden.clone(), width.clone()),
            )?;
            let bias = load_weight(graph, weights, draft, corrector_affine(CorrectorSite::Token, AffinePart::Bias))?;
            (token, bias.reshape(&[&one, width]), 1)
        }
        None => (
            embedded.r1,
            GraphConstant::absent(graph, constants)?.tensor().reshape(&[&Dim::from(0), width]),
            0,
        ),
    };

    // The mask rows' pass rows.
    let pass_rows = graph
        .enqueue(
            entries.pass_norm,
            &[
                ("M", Dim::from(path.rows)),
                ("O", Dim::from(outputs)),
                ("D", hidden.clone()),
                ("PT", Dim::from(1)),
                ("NM", hidden.clone()),
            ],
            readout_features_rows::WorkflowArgs {
                hidden: (&path.residual.reshape(&[&Dim::from(path.rows), &one, hidden])).into(),
                norm: path.output_norm.into(),
                out_rows: path.head_rows.into(),
                epsilon,
            },
        )?
        .value;
    let pass = project(
        graph,
        weights,
        constants,
        entries.pass,
        (draft, CorrectorSite::Pass),
        (&pass_rows).into(),
        (Dim::from(outputs), hidden.clone(), width.clone()),
    )?;

    // The nodes.
    let pass_bias = load_weight(graph, weights, draft, corrector_affine(CorrectorSite::Pass, AffinePart::Bias))?;
    let feature_bias =
        load_weight(graph, weights, draft, corrector_affine(CorrectorSite::FeatureDown, AffinePart::Bias))?;
    let slot_embedding = load_weight(graph, weights, draft, corrector_weight(CorrectorWeight::SlotEmbedding))?;
    let rank_embedding = load_weight(graph, weights, draft, corrector_weight(CorrectorWeight::RankEmbedding))?;
    let (slots_dim, mask_dim, top_k_dim) = (Dim::from(slots), Dim::from(mask_rows), Dim::from(top_k));
    let node_shape = [&slots_dim, &mask_dim, &top_k_dim, width];
    let mut stream = graph
        .enqueue(
            entries.nodes,
            &[
                ("S", slots_dim.clone()),
                ("J", mask_dim.clone()),
                ("K", top_k_dim.clone()),
                ("H", width.clone()),
                ("TB", Dim::from(token_biases)),
            ],
            corrector_nodes::WorkflowArgs {
                token: (&token.reshape(&node_shape)).into(),
                token_bias: (&token_bias).into(),
                pass: (&pass.reshape(&[&slots_dim, &mask_dim, width])).into(),
                pass_bias: (&pass_bias).into(),
                feature: (&feature.reshape(&node_shape)).into(),
                feature_bias: (&feature_bias).into(),
                slot: (&slot_embedding).into(),
                rank: (&rank_embedding).into(),
            },
        )?
        .value;

    // The corrector's layers.
    let attention_bias = load_weight(graph, weights, draft, corrector_weight(CorrectorWeight::AttentionBias))?;
    let (nodes_dim, node_rows_dim) = (Dim::from(nodes), Dim::from(node_rows));
    for (index, (&[attention_epsilon, mlp_epsilon], stages)) in
        scalars.layers.iter().zip(&entries.layers).enumerate()
    {
        let scope =
            WeightScope::CorrectorLayer(u32::try_from(index).map_err(|_| "corrector layer index exceeds u32")?);
        let attention_norm = load_weight(graph, weights, scope, corrector_weight(CorrectorWeight::AttentionNorm))?;
        let normed = norm(
            graph,
            constants,
            stages.attention_norm,
            &stream,
            &attention_norm,
            (node_rows, width),
            attention_epsilon.f32(),
        )?;
        let qkv = project(
            graph,
            weights,
            constants,
            stages.attention_input,
            (scope, CorrectorSite::AttentionInput),
            (&normed).into(),
            (node_rows_dim.clone(), width.clone(), width * 3),
        )?;
        let input_bias =
            load_weight(graph, weights, scope, corrector_affine(CorrectorSite::AttentionInput, AffinePart::Bias))?;
        let attended = graph
            .enqueue(
                entries.attention,
                &[
                    ("S", slots_dim.clone()),
                    ("N", nodes_dim.clone()),
                    ("NH", heads.clone()),
                    ("W", head_width.clone()),
                ],
                corrector_attention::WorkflowArgs {
                    qkv: (&qkv.reshape(&[&slots_dim, &nodes_dim, &three, heads, &head_width])).into(),
                    input_bias: (&input_bias.reshape(&[&three, heads, &head_width])).into(),
                    bias: (&attention_bias).into(),
                },
            )?
            .value;
        let output = project(
            graph,
            weights,
            constants,
            stages.attention_output,
            (scope, CorrectorSite::AttentionOutput),
            (&attended.reshape(&[&node_rows_dim, width])).into(),
            (node_rows_dim.clone(), width.clone(), width.clone()),
        )?;
        stream = finish(
            graph,
            weights,
            constants,
            stages.attention_residual,
            (scope, CorrectorSite::AttentionOutput),
            &output,
            Some(&stream),
            (0, 0.0.into()),
            (node_rows_dim.clone(), width.clone()),
        )?;
        let mlp_norm = load_weight(graph, weights, scope, corrector_weight(CorrectorWeight::MlpNorm))?;
        let normed = norm(graph, constants, stages.mlp_norm, &stream, &mlp_norm, (node_rows, width), mlp_epsilon.f32())?;
        let up = project(
            graph,
            weights,
            constants,
            stages.up,
            (scope, CorrectorSite::Up),
            (&normed).into(),
            (node_rows_dim.clone(), width.clone(), mlp.clone()),
        )?;
        let activated = finish(
            graph,
            weights,
            constants,
            stages.activated,
            (scope, CorrectorSite::Up),
            &up,
            None,
            (1, 0.0.into()),
            (node_rows_dim.clone(), mlp.clone()),
        )?;
        let down = project(
            graph,
            weights,
            constants,
            stages.down,
            (scope, CorrectorSite::Down),
            (&activated).into(),
            (node_rows_dim.clone(), mlp.clone(), width.clone()),
        )?;
        stream = finish(
            graph,
            weights,
            constants,
            stages.down_residual,
            (scope, CorrectorSite::Down),
            &down,
            Some(&stream),
            (0, 0.0.into()),
            (node_rows_dim.clone(), width.clone()),
        )?;
    }
    let output_norm = load_weight(graph, weights, draft, corrector_weight(CorrectorWeight::OutputNorm))?;
    let normed = norm(graph, constants, entries.output_norm, &stream, &output_norm, (node_rows, width), scalars.output.f32())?;

    // Each slot's anchor state: the fused feature of its last entry row (the
    // row that selected its anchor), its context projection and the anchor
    // norm.
    let anchor_dims = [
        ("M", path.entry_rows.clone()),
        ("O", slots_dim.clone()),
        ("D", hidden.clone()),
        ("PT", Dim::from(1)),
        ("NM", hidden.clone()),
    ];
    let anchor_rows = graph.input_for(entries.anchor_rows, "out_rows", &anchor_dims)?;
    let fused = graph
        .enqueue(
            entries.anchor_rows,
            &anchor_dims,
            readout_features_rows::WorkflowArgs {
                hidden: (&path.conditioning.reshape(&[path.entry_rows, &one, hidden])).into(),
                norm: path.fusion_norm.into(),
                out_rows: anchor_rows.tensor().into(),
                epsilon,
            },
        )?
        .value;
    let context = project(
        graph,
        weights,
        constants,
        entries.context,
        (draft, CorrectorSite::Context),
        (&fused).into(),
        (slots_dim.clone(), hidden.clone(), width.clone()),
    )?;
    let context = finish(
        graph,
        weights,
        constants,
        entries.context_bias,
        (draft, CorrectorSite::Context),
        &context,
        None,
        (0, 0.0.into()),
        (slots_dim.clone(), width.clone()),
    )?;
    let anchor_norm = load_weight(graph, weights, draft, corrector_weight(CorrectorWeight::AnchorNorm))?;
    let anchor_state =
        norm(graph, constants, entries.anchor_norm, &context, &anchor_norm, (slots, width), scalars.anchor.f32())?;

    // The factored vectors: each node's [x ; a ; x · a] through the factor
    // input, then the unit-length outgoing and incoming heads; the anchor
    // head of each slot's anchor state.
    let joined = graph
        .enqueue(
            entries.factors,
            &[("S", slots_dim.clone()), ("N", nodes_dim.clone()), ("H", width.clone())],
            corrector_factors::WorkflowArgs {
                nodes: (&normed.reshape(&[&slots_dim, &nodes_dim, width])).into(),
                anchor: (&anchor_state).into(),
            },
        )?
        .value;
    let factor_input = project(
        graph,
        weights,
        constants,
        entries.factor_input,
        (draft, CorrectorSite::FactorInput),
        (&joined.reshape(&[&node_rows_dim, &(width * 3)])).into(),
        (node_rows_dim.clone(), width * 3, width.clone()),
    )?;
    let factored = finish(
        graph,
        weights,
        constants,
        entries.factor_activated,
        (draft, CorrectorSite::FactorInput),
        &factor_input,
        None,
        (1, 0.0.into()),
        (node_rows_dim.clone(), width.clone()),
    )?;
    let head = |graph: &mut G,
                weights: &mut Vec<(WeightPort, NativePort)>,
                constants: &mut Vec<GraphConstant>,
                entry: G::Binding<'a, project_rows::Entry>,
                site: CorrectorSite,
                source: &WorkflowTensor,
                count: &Dim| {
        let projected =
            project(graph, weights, constants, entry, (draft, site), source.into(), (count.clone(), width.clone(), factor.clone()))?;
        finish(
            graph,
            weights,
            constants,
            entries.unit,
            (draft, site),
            &projected,
            None,
            (2, scalars.vector.f32()),
            (count.clone(), factor.clone()),
        )
    };
    let outgoing = head(graph, weights, constants, entries.outgoing, CorrectorSite::Outgoing, &factored, &node_rows_dim)?;
    let incoming = head(graph, weights, constants, entries.incoming, CorrectorSite::Incoming, &factored, &node_rows_dim)?;
    let anchor = head(graph, weights, constants, entries.anchor, CorrectorSite::Anchor, &anchor_state, &slots_dim)?;

    // The greedy walk of each slot's lattice.
    let factor_shape = [&slots_dim, &mask_dim, &top_k_dim, factor];
    let mut selection = path.result.reshape(&[path.proposals, slots, 2]);
    graph.enqueue(
        entries.path,
        &[
            ("S", slots_dim.clone()),
            ("J", mask_dim.clone()),
            ("K", top_k_dim.clone()),
            ("F", factor.clone()),
            ("P", Dim::from(path.proposals)),
        ],
        corrector_path::WorkflowArgs {
            candidates: (&candidates.reshape(&[slots, mask_rows, top_k, 2])).into(),
            incoming: (&incoming.reshape(&factor_shape)).into(),
            outgoing: (&outgoing.reshape(&factor_shape)).into(),
            anchor: (&anchor).into(),
            selection: (&mut selection).into(),
        },
    )?;
    Ok((anchor_rows, landed))
}

type PreparedDraftGraph = DraftGraphParts<NativeGraphPlan>;

pub struct PreparedDraftGraphs {
    classes: BTreeMap<DraftGraphClass, PreparedDraftGraph>,
    family: NativeGraphFamily,
    /// The block pass's visible spans.
    segments: u64,
}

pub(crate) struct BoundDraftGraphs {
    prepared: Rc<PreparedDraftGraphs>,
    bound: BTreeMap<DraftGraphClass, BoundNativeGraphPlan>,
    constants: Vec<Tensor>,
    /// The block embedding as the block's tokens gather it.
    embedding: crate::device_table::ResidentTable,
}

impl PreparedDraftGraphs {
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        distinct_storage_bytes(
            self.classes
                .values()
                .flat_map(|graph| graph.constants.iter()),
        )
    }

    /// Every class's graph, sealed into the layout its topology is certified
    /// into (`certify_draft_family`), so a load holds what its assessment
    /// charged. `landing`: of the block's token table, when it has landing
    /// slots.
    pub(crate) fn prepare(
        target_device: &Device,
        draft: &AttestedDraft,
        landing: Option<&AttestedLanding>,
        load: &ModelLoadPlan,
        plan: &DraftProgramPlan,
        geometry: &DraftGeometry<'_>,
        classes: impl IntoIterator<Item = DraftGraphClass>,
    ) -> Result<Self, SubmitError> {
        let classes = classes.into_iter().collect::<Vec<_>>();
        let split = DraftSplit::of(target_device.backend(), load, plan, geometry)
            .map_err(|error| invalid(error.to_string()))?;
        let (resources, layouts) = certify_draft_family::<NativeGraphLayout>(&split, &classes)
            .map_err(|error| invalid(error.to_string()))?;
        let numbers = split.numbers();
        let mut prepared = BTreeMap::new();
        for class in classes {
            let class_error =
                |error: GraphError| invalid(format!("draft graph class {class:?}: {error}"));
            let parts = split
                .build(
                    target_device.native_graph_with_layout(&layouts[&class]),
                    DraftGraphEntries::prepared(draft, landing),
                    &numbers,
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
        let plans = prepared.values().map(|graph| (false, graph.plan.clone()));
        let family = super::graph::serial_family("drafter", plans, &resources.graphs).map_err(device)?;
        Ok(Self {
            classes: prepared,
            family,
            segments: geometry.segments,
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
                            .slice_leading(0, block.readout_vocabulary.fixed())
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
            embedding: resident.embedding.table().clone(),
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
/// draft's own), the target's projection as it is stored (packed, or its
/// planes), or a drafter weight.
fn resident_draft_weight(
    resident: &ResidentHead,
    role: WeightRole,
) -> Result<&ResidentWeight, SubmitError> {
    match (role.scope, role.kind) {
        (WeightScope::Target | WeightScope::Draft, WeightKind::Embedding) => {
            Ok(&resident.embedding)
        }
        (WeightScope::Target | WeightScope::Draft, WeightKind::Output) => match &resident.output {
            ResidentOutput::Packed(weight) => Ok(weight),
            ResidentOutput::Progressive(_) => Err(invalid(
                "the draft reads the packed projection of a load that stores its planes",
            )),
        },
        (WeightScope::Target, WeightKind::OutputPlane(plane)) => match &resident.output {
            ResidentOutput::Progressive(planes) => Ok(planes.plane(plane)),
            ResidentOutput::Packed(_) => Err(invalid(
                "the draft reads the planes of a load that stores its packed projection",
            )),
        },
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

/// Fresh rows are contiguous within a slot. Intersect its causal or
/// bidirectional span with the layer's strict-distance window (|q-k| < W).
fn draft_fresh_span(span: [i32; 2], row: usize, window: Option<u64>) -> [i32; 2] {
    let Some(window) = window else { return span };
    let radius = window.saturating_sub(1).min(i32::MAX as u64) as i64;
    let query = row as i64;
    [
        i64::from(span[0]).max(query - radius) as i32,
        i64::from(span[1]).min(query + radius + 1) as i32,
    ]
}

impl PassControls {
    fn new(
        pass: &TargetBatchUpload<'_>,
        domain: usize,
        rows: usize,
        segments: usize,
        block: Option<BlockAttention>,
        window: Option<u64>,
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
            let span = match block {
                Some(BlockAttention::Causal) => history.fresh[row],
                Some(BlockAttention::Bidirectional) => pass.segments[pass.row_slots[row] as usize],
                None => [0, 0],
            };
            draft_fresh_span(span, row, window)
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
pub(crate) fn widen(bytes: &[u8], dtype: ActivationDType) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .flat_map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let value = match dtype {
                ActivationDType::BF16 => f32::from_bits(u32::from(bits) << 16),
                ActivationDType::F16 => seismic::f16_to_f32(bits),
            };
            value.to_le_bytes()
        })
        .collect()
}

pub struct NativeDraftProgram {
    definition: ModelDefinition,
    /// Immutable normalized per-layer window bounds, resolved once at load.
    windows: Vec<Option<u64>>,
    graphs: BoundDraftGraphs,
    waiter: CompletionWaiter,
}

impl NativeDraftProgram {
    /// Visit the graph constants the program binds.
    pub(crate) fn visit_constants(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        self.graphs.constants.iter().for_each(visit);
    }

    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    pub(crate) fn new(
        definition: ModelDefinition,
        graphs: BoundDraftGraphs,
    ) -> Result<Self, SubmitError> {
        let draft = definition
            .draft
            .as_ref()
            .ok_or_else(|| invalid("the draft program's definition has no draft"))?;
        let windows = (0..draft.blocks.len())
            .map(|index| {
                let paired = operators::draft::draft_block(draft, index)
                    .map_err(|error| invalid(error.to_string()))?;
                match paired.mixer {
                    Mixer::Attention(attention) => {
                        Ok(match operators::attention::fresh_rows(attention) {
                            operators::attention::FreshRows::Own(ProjectedSource {
                                domain: TokenDomain::Window { tokens },
                                ..
                            }) => Some(*tokens),
                            _ => None,
                        })
                    }
                    _ => Err(invalid("draft layers attend")),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let waiter = CompletionWaiter::spawn().map_err(device)?;
        Ok(Self {
            definition,
            windows,
            graphs,
            waiter,
        })
    }

    fn queue(
        &self,
        core: &HeadLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: NativeGraphOutputLease,
    ) -> Result<(seismic::NativeGraphCompletion, HeadOutput), SubmitError> {
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
            proposals: steps as u64,
            shaped,
            windowed: block.is_some() || core.windowed(),
            device: matches!(
                core.conditioning(),
                crate::HeadConditioning::Features { .. }
            ),
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
            (
                DraftConditioning::Features(port),
                crate::HeadConditioning::Features {
                    current: features,
                    previous: None,
                },
            ) => {
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
            let controls = PassControls::new(&entry, domain, entry_rows, 1, None, None)?;
            write_controls(&mut active, &ports.controls, &controls)?;
        }
        if let (Some(ports), Some(block)) = (&graph.block, &block) {
            let block_rows = slots * draft.block_rows(steps as u64) as usize;
            let segments = self.graphs.prepared.segments as usize;
            // The block's tokens gather by table row: of a streamed table,
            // by the slots their rows are brought to. Padding rows past the
            // block's name no row (-1) and read zeros.
            let tokens = (0..block_rows)
                .map(|row| {
                    block
                        .tokens
                        .get(row)
                        .copied()
                        .filter(|_| row < block.actual_rows)
                        .unwrap_or(-1)
                })
                .collect::<Vec<_>>();
            let rows = self.graphs.embedding.indices(&tokens).map_err(invalid)?;
            active
                .write_input(
                    &ports.tokens,
                    &i32_bytes(rows.iter().flat_map(|row| [*row, 0])),
                )
                .map_err(device)?;
            for (index, ((layer, &domain), &attention)) in ports
                .layers
                .iter()
                .zip(&layer_domains)
                .zip(&draft.block_attention)
                .enumerate()
            {
                let controls = PassControls::new(
                    block,
                    domain,
                    block_rows,
                    segments,
                    Some(attention),
                    self.windows[index],
                )?;
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
            let slot_rows = draft.block_rows(steps as u64);
            let rows = match &ports.anchor_rows {
                // A corrector projects every mask row of every slot,
                // slot-major.
                Some(_) => (0..slots as u64)
                    .flat_map(|slot| {
                        (0..draft.max_proposals())
                            .map(move |mask| slot * slot_rows + draft.proposal_row(mask))
                    })
                    .map(|row| i32::try_from(row).map_err(|_| invalid("draft row exceeds i32")))
                    .collect::<Result<Vec<i32>, SubmitError>>()?,
                None => (0..steps)
                    .flat_map(|step| (0..slots).map(move |slot| (step, slot)))
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
                    .collect::<Result<Vec<i32>, SubmitError>>()?,
            };
            active
                .write_input(&ports.head_rows, &i32_bytes(rows))
                .map_err(device)?;
            let words = ports.readout_vocabulary.fixed().div_ceil(32) as usize;
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
                            .unwrap_or(-1),
                        0,
                    ]
                });
                active
                    .write_input(anchors, &i32_bytes(anchors_of))
                    .map_err(device)?;
            }
            // Each slot's last entry row, whose conditioning selected its
            // anchor; padding slots read row 0.
            if let Some(anchor_rows) = &ports.anchor_rows {
                let mut last = vec![0_i32; slots];
                for (row, &slot) in entry.row_slots[..entry.actual_rows].iter().enumerate() {
                    let slot = usize::try_from(slot)
                        .map_err(|_| invalid("draft entry row has no slot"))?;
                    *last
                        .get_mut(slot)
                        .ok_or_else(|| invalid("draft entry row's slot exceeds the class"))? =
                        i32::try_from(row).map_err(|_| invalid("draft entry row exceeds i32"))?;
                }
                active
                    .write_input(anchor_rows, &i32_bytes(last))
                    .map_err(device)?;
            }
        }
        // A corrector's candidates are table rows the device selects, a
        // landed read of a table with landing slots.
        let landed = graph
            .block
            .as_ref()
            .and_then(|block| block.landed.as_ref())
            .map(|ports| ports.queue(&self.graphs.embedding, &mut bindings))
            .transpose()
            .map_err(invalid)?;
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
                let block = graph
                    .block
                    .as_ref()
                    .ok_or_else(|| invalid("a drafting draft graph has no block"))?;
                Ok::<_, SubmitError>(HeadSelections {
                    tokens: owner
                        .tensor(&graph.output)
                        .ok_or_else(|| invalid("draft graph omitted its selections"))?,
                    confidence: block
                        .confidence
                        .as_ref()
                        .map(|confidence| {
                            owner
                                .tensor(confidence)
                                .ok_or_else(|| invalid("draft graph omitted its confidence"))
                        })
                        .transpose()?,
                })
            })
            .transpose()?;
        Ok((
            completion,
            HeadOutput {
                selections,
                landed_reads: landed.into_iter().map(SelectedRead::submitted).collect(),
            },
        ))
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
    if let Some(visible) = &ports.visible {
        active
            .write_input(visible, &controls.visible)
            .map_err(device)?;
    }
    if let Some(fresh) = &ports.fresh {
        active.write_input(fresh, &controls.fresh).map_err(device)?;
    }
    active
        .write_input(&ports.destinations, &controls.destinations)
        .map_err(device)
}

impl HeadProgram for NativeDraftProgram {
    type Submission =
        DeviceSubmission<HeadLaunchCore, NativeGraphWorkspaceLease, HeadOutput>;

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
        let (completion, output) = match queued {
            Ok(queued) => queued,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace, _) = launch.into_submission_parts();
        Ok(DeviceSubmission::new(
            self.waiter.completion(vec![completion]),
            core,
            workspace,
            output,
        ))
    }
}

#[cfg(test)]
mod window_tests {
    use super::draft_fresh_span;

    #[test]
    fn fresh_windows_clip_both_sides_without_crossing_slots() {
        for start in [0, 9] {
            for width in [1, 2, 4, 32] {
                for row in start..start + 8 {
                    for causal in [false, true] {
                        let end = if causal { row + 1 } else { start + 8 };
                        let span = draft_fresh_span([start as i32, end as i32], row, Some(width));
                        for key in 0usize..24 {
                            let expected =
                                key >= start && key < end && row.abs_diff(key) < width as usize;
                            assert_eq!(key as i32 >= span[0] && (key as i32) < span[1], expected);
                        }
                    }
                }
            }
        }
        assert_eq!(draft_fresh_span([9, 17], 12, None), [9, 17]);
    }
}
