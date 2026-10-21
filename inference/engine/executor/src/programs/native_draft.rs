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
//!
//! Selections are step-major: proposal `k` of slot `s` is result row
//! `k · slots + s`.

use super::{DeviceSubmission, HeadOutput, HeadProgram, SelectedRows};
use crate::operators::{self, InputForm, Join, Mixer, SublayerInput};
use crate::{
    completion::CompletionWaiter,
    native::AttestedDraft,
    operators::attention::graph::{
        self as attention_graph, attention_weights, AttentionBlock, AttentionBounds,
        AttentionControlPorts, AttentionGraphEntries, AttentionHistoryEntries, AttentionWeights,
        CheckedAttentionEntries, ProjectedRows,
    },
    operators::dense_ffn::graph::{self as dense_graph, CheckedDenseEntries, DenseGraphEntries},
    programs::{
        graph::readout::{self, shapes, write_selection_rows, SelectionPorts},
        graph::{draft::GraphDraft, GraphError, RowForm},
        native_constants::{
            distinct_storage_bytes, CheckedGraphFamilyResources, CheckedGraphResources,
            ConstantTensors, GraphConstant,
        },
        native_target_graph::{
            weight, weight_scale_port, GatheredTable, PackingBound, RowDirectory, WeightPort,
        },
    },
    DeviceError, DraftProgramPlan, HeadLaunchCore, InvariantError,
    ModelLoadPlan, NativeGraphOutputLease, NativeGraphWorkspaceLease, ResidentHead, ResidentOutput,
    ResidentWeight, ResourceLimits, StateStorePlan, SubmitError, ValidatedHeadLaunch,
};
use magnitude_batching::{row_class, TargetBatchUpload};
use magnitude_family_contracts::{
    ActivationDType, AffinePart, BlockAttention, CandidateCorrector, CorrectorSite,
    CorrectorWeight, DraftDefinition, DraftEmbedding, DraftMethod, HistoryDomain, KeyValue,
    ModelDefinition, SublayerIndex, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    corrector_attention, corrector_factors, corrector_features, corrector_nodes, corrector_path,
    corrector_rows, dense_output, draft_convolve_input, draft_convolve_residual, draft_gated_rows, draft_path_step,
    draft_top_k, embedding_rows, project_rows, readout_features_rows, sample_rows, shape_rows,
    widen_rows,
};
use magnitude_state::LayerRef;
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphClassSlice,
    NativeGraphFamily, NativeGraphLayout, NativeGraphMetadata, NativeGraphPlan, NativePort, Tensor,
    WorkflowTensor,
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
            Ok(match attention.key_value {
                KeyValue::Owned {
                    domain: HistoryDomain::Window { tokens },
                    ..
                } => Some(tokens),
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
    pub head: readout::ProjectionEntries<'a, G>,
    pub shape: G::Binding<'a, shape_rows::Entry>,
    pub sample: G::Binding<'a, sample_rows::Entry>,
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
    fn prepared(draft: &'a AttestedDraft) -> Self {
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
            head: readout::ProjectionEntries::prepared(&draft.head),
            shape: &draft.shape,
            sample: &draft.sample,
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
/// metadata-only (checked) graphs.
struct CheckedDraftEntries {
    layers: Vec<(
        CheckedAttentionEntries,
        CheckedAttentionEntries,
        CheckedDenseEntries,
    )>,
    embedding: [(&'static str, Element); 2],
    head: readout::CheckedProjectionEntries,
    selection: [(&'static str, Element); 0],
    widen: [(&'static str, Element); 1],
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

struct CheckedMarkovEntries {
    embedding: [(&'static str, Element); 2],
    projection: [(&'static str, Element); 2],
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
                        CheckedAttentionEntries::new(block.attention, false),
                        CheckedAttentionEntries::new(block.injection, false),
                        CheckedDenseEntries::new(block.feed_forward),
                    )
                })
                .collect(),
            embedding: [("EW", plan.embedding().table), ("A", activation)],
            head: readout::CheckedProjectionEntries::new(
                plan.projection(),
                plan.output_norm(),
                activation,
                false,
            ),
            selection: [],
            widen: [("A", activation)],
            markov: plan.markov().map(|markov| CheckedMarkovEntries {
                embedding: [("EW", markov.embedding), ("A", activation)],
                projection: [("DW", markov.projection), ("A", activation)],
            }),
            convolved: plan
                .convolutions()
                .map(|binding| CheckedConvolvedEntries::new(plan, binding)),
            selector: plan
                .selector()
                .map(|binding| CheckedSelectorEntries::new(plan, binding)),
            corrector: plan
                .corrector()
                .map(|binding| CheckedCorrectorEntries::new(plan, binding)),
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
            head: self.head.entries(),
            shape: &self.selection[..],
            sample: &self.selection[..],
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
    certify_draft_family(backend, load, plan, geometry, &classes).map(|(resources, _)| resources)
}

/// The family's storage and each class's layout. Every class field but the
/// entry rows fixes a graph's topology, as does the attention form those
/// rows take (`RowForm`): only the injection pass follows them, the block
/// pass's rows being its slots' and proposals'. So each topology is one
/// template, built at its largest entry rows and certified over every
/// class's, and each class seals into its topology's layout
/// (`PreparedDraftGraphs::prepare`). The injection pass's only constants
/// that follow its rows are empty (an attention prefill's `[rows, 0]`
/// bitmap), so the template's constants are every class's.
fn certify_draft_family(
    backend: BackendName,
    load: &ModelLoadPlan,
    plan: &DraftProgramPlan,
    geometry: &DraftGeometry<'_>,
    classes: &[DraftGraphClass],
) -> Result<
    (
        CheckedGraphResources,
        BTreeMap<DraftGraphClass, NativeGraphLayout>,
    ),
    GraphError,
> {
    let checked = CheckedDraftEntries::new(plan);
    let mut groups: BTreeMap<(u64, u64, bool, bool, bool, RowForm), Vec<DraftGraphClass>> =
        BTreeMap::new();
    for &class in classes {
        if groups.values().any(|group| group.contains(&class)) {
            return Err("draft graph class is duplicated".into());
        }
        groups
            .entry((
                class.slots,
                class.proposals,
                class.shaped,
                class.windowed,
                class.device,
                RowForm::of(class.entry_rows),
            ))
            .or_default()
            .push(class);
    }
    if groups.is_empty() {
        return Err("draft graph family has no classes".into());
    }
    let mut family = CheckedGraphFamilyResources::new();
    let mut layouts = BTreeMap::new();
    for group in groups.into_values() {
        let largest = *group
            .iter()
            .max_by_key(|class| class.entry_rows)
            .expect("a topology holds a class");
        let context = || format!("draft graph class {largest:?}");
        let parts = draft_graph(
            NativeGraphMetadata::new_template(backend),
            checked.entries()?,
            load,
            geometry,
            largest,
        )
        .map_err(|error| error.context(context()))?;
        #[cfg(test)]
        if let Some(block) = &parts.block {
            assert_eq!(
                block.readout_vocabulary, geometry.definition.decoder.vocabulary,
                "separate draft graph must retain every artifact vocabulary row"
            );
        }
        let template = parts
            .plan
            .seal_template()
            .map_err(|error| GraphError::from(error).context(context()))?;
        let slices = group
            .iter()
            .map(|class| {
                NativeGraphClassSlice::new()
                    .dimension(ENTRY_ROWS, [class.entry_rows])
                    .scoped(INJECTION, "M", [class.entry_rows])
            })
            .collect::<Vec<_>>();
        let layout = template
            .certify(&slices)
            .map_err(|error| GraphError::from(error).context(context()))?;
        family.include(layout.storage_bytes(), parts.constants);
        for &class in &group {
            family.include_plan(layout.plan_heap())?;
            layouts.insert(class, layout.clone());
        }
    }
    Ok((family.finish()?, layouts))
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
    let (resources, layouts) = certify_draft_family(backend, load, plan, geometry, classes)?;
    let checked = CheckedDraftEntries::new(plan);
    let mut exact = CheckedGraphFamilyResources::new();
    for &class in classes {
        let parts = draft_graph(
            NativeGraphMetadata::new(backend),
            checked.entries()?,
            load,
            geometry,
            class,
        )?;
        let charged = parts
            .plan
            .seal_with_layout(&layouts[&class])
            .map_err(|error| GraphError::from(error).context(format!("draft class {class:?}")))?;
        if charged != layouts[&class].storage_bytes() {
            return Err(format!("draft class {class:?} charged a different layout").into());
        }
        exact.include(
            seismic::NativeGraphStorageBytes {
                workspace: 0,
                output: 0,
                upload: 0,
            },
            parts.constants,
        );
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
    /// A corrector's candidates `[slots · mask_rows, top_k, 2]` (token,
    /// status), exported: the block's token-table rows its device work
    /// selects, which the host observes (`SelectedRows`).
    candidates: Option<WorkflowTensor>,
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
    let skippable = skippable_layers(draft)?;
    for index in 0..draft.blocks.len() {
        let paired =
            operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
        let Mixer::Attention(attention) = paired.mixer else {
            return Err("draft layers attend".into());
        };
        if skippable[index] && !class.windowed {
            layer_weights.push(None);
            continue;
        }
        let shape =
            operators::attention::shape(hidden, attention).map_err(|error| error.to_string())?;
        let scope = draft_sublayer(index, 0)?;
        let bounds = AttentionBounds::planned(
            &mut graph,
            load,
            scope,
            attention,
            &shape,
            class.slots > 0,
            &mut weights,
            &mut constants,
        )?;
        let injection = attention_weights(
            &shape,
            attention,
            InputForm::SINGLE,
            class.slots > 0,
            false,
            bounds,
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
            // The draft's classes list no history row tiles.
            history_tiles: 0,
            shape,
            operator,
            epsilon,
            head_epsilon: epsilon,
            post_norm_epsilon: 0.0,
            post_norm_scale: 1.0,
            activation,
            inject_only,
            causal_cache: false,
            selection: None,
        }
    }

    // Injection: each entry row appends its context K/V at its destination.
    // The host writes the rows (an input of the first injection's
    // projection; every layer's projection reads the same rows), or a
    // device-conditioned class widens its bound target features.
    let (first, (_, first_shape, first_weights)) = layer_weights
        .iter()
        .enumerate()
        .find_map(|(index, layer)| layer.as_ref().map(|layer| (index, layer)))
        .ok_or("a draft has no injected layers")?;
    let first_bounds = first_weights.bounds.project_dimensions(true);
    // Only the injection pass follows the entry rows: its rows are a
    // topology's class dimension (`certify_draft_family`).
    graph.set_class_scope(Some(INJECTION));
    let (conditioning, conditioning_rows) = if class.device {
        let features =
            graph.port_with_class_extent(activation, &[class.entry_rows, hidden], 0, ENTRY_ROWS)?;
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
            &[
                first_shape
                    .key_value_dimensions(class.entry_rows)
                    .as_slice(),
                operators::attention::UNSCALED_PROJECT.as_slice(),
                first_bounds.as_slice(),
                &InputForm::SINGLE.dimensions(first_shape.hidden),
            ]
            .concat(),
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
        let join = Join::single(Some(ENTRY_ROWS));
        let (mixed, state, controls) = attention_graph::attention(
            &mut graph,
            layer.injection,
            weights_of,
            &mut constants,
            SublayerInput::single(&conditioning_rows),
            &conditioning_rows,
            &join,
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
    graph.set_class_scope(None);
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

    // The block pass. Its unscaled weights bind the graph's absent operand
    // (`GraphConstant::absent`).
    let slots = class.slots;
    let proposals = class.proposals;
    let block_rows = draft.block_rows(proposals);
    let rows = slots * block_rows;
    let corrector = match (&draft.method, &entries.corrector) {
        (DraftMethod::LiLiCorr { corrector }, Some(stages)) => Some((corrector.as_ref(), stages)),
        (DraftMethod::LiLiCorr { .. }, None) | (_, Some(_)) => {
            return Err("draft method and corrector entries disagree".into());
        }
        _ => None,
    };
    // The selections, step-major; a corrector projects every mask row of
    // every slot (slot-major), the rest each proposing row.
    let selected = proposals * slots;
    let mask_rows = draft.max_proposals();
    let outputs = match corrector {
        Some(_) => slots * mask_rows,
        None => selected,
    };
    // The block's tokens are the host's (`BoundDraftGraphs::embedding`).
    let mut table = GatheredTable::of(
        &mut graph,
        load,
        match &draft.embedding {
            DraftEmbedding::Target => WeightScope::Target,
            DraftEmbedding::Own(_) => WeightScope::Draft,
        },
        WeightKind::Embedding,
        &mut weights,
    )?;
    // The empty directory: every table this graph gathers, it gathers by
    // the host's rows or the rows of a table held whole (`gathered_table`).
    let direct = table.by_rows(&mut graph, &mut constants)?;
    let embedding_dims = [
        ("M", rows),
        ("V", table.rows),
        ("D", hidden),
        ("R", direct.shape[0]),
        ("F", direct.shape[1]),
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
                scale: draft.embedding_scale as f32,
                normalize: 0,
                epsilon,
            },
        )?
        .r1;
    let mut layers = Vec::with_capacity(draft.blocks.len());
    // A convolved backbone's unfused layers read their normed rows through
    // the identity row map of the block rows.
    let convolved = match (&draft.convolutions, &entries.convolved) {
        (Some(convolutions), Some(stages)) => {
            let identity = GraphConstant::identity(&mut graph, rows)?;
            let rows_map = identity.port().tensor().clone();
            constants.push(identity);
            Some((stages, convolutions.kernel, convolutions.group, rows_map))
        }
        (None, None) => None,
        _ => return Err("draft convolutions and entries disagree".into()),
    };
    for (index, (layer_weight, layer)) in layer_weights.iter().zip(&entries.layers).enumerate() {
        // A drafting class injects every layer.
        let (operator, shape, injection_weights) = layer_weight
            .as_ref()
            .ok_or("a drafting class skips a layer's injection")?;
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
            input_norm: Some(input_norm),
            query: injection_weights.query.clone(),
            gate: injection_weights.gate.clone(),
            key: injection_weights.key.clone(),
            value: injection_weights.value.clone(),
            query_norm: injection_weights.query_norm.clone(),
            key_norm: injection_weights.key_norm.clone(),
            sinks: injection_weights.sinks.clone(),
            output: injection_weights.output.clone(),
            post_norm: None,
            bounds: injection_weights.bounds.clone(),
            indexer: None,
        };
        let join = Join::single(None);
        let (mixed, state, controls) = attention_graph::attention(
            &mut graph,
            layer.attention,
            &block_weights,
            &mut constants,
            SublayerInput::single(&residual),
            &residual,
            &join,
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
    let projection_scope = if draft.output.is_some() {
        WeightScope::Draft
    } else {
        WeightScope::Target
    };
    let readout_vocabulary = vocabulary;
    let mut leading = Vec::new();
    // Full draft logits retain the canonical packed vocabulary prefix.
    // A scope a separate draft projects through is never stored as planes
    // (`ModelLoadPlan::with_readout_candidates`).
    let plans = readout::HeadPlans::of(load, projection_scope)?.selected();
    let mut prefix = |plan: &crate::WeightPlan| -> Result<NativePort, GraphError> {
        let mut shape = plan.shape.clone();
        shape[0] = readout_vocabulary;
        let port = graph.port(plan.resident, &shape)?;
        leading.push((plan.role, port.clone()));
        Ok(port)
    };
    let (projection, head_rows) = match plans {
        readout::HeadPlans::Packed(plan) => {
            let weight = prefix(plan)?;
            let (scale, extent) = weight_scale_port(
                &mut graph,
                load,
                WeightRole {
                    scope: projection_scope,
                    kind: WeightKind::Output,
                },
                &mut weights,
                &mut constants,
            )?;
            let head_rows = entries.head.raw_packed_out_rows(
                &mut graph,
                &[
                    ("M", rows),
                    ("O", outputs),
                    ("V", readout_vocabulary),
                    ("D", hidden),
                    ("WS", extent),
                    ("PT", 1),
                    ("NM", hidden),
                ],
            )?;
            (
                readout::ProjectionPorts::Packed {
                    weight,
                    scale,
                    scale_extent: extent,
                },
                head_rows,
            )
        }
        readout::HeadPlans::Progressive(_) | readout::HeadPlans::Qualified { .. } => {
            return Err("draft full logits require the canonical packed projection".into());
        }
    };
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
            rows,
            projected: outputs,
            selected: 0,
            vocabulary: readout_vocabulary,
            hidden,
            columns: readout_vocabulary,
        },
        draft.logit_softcap.unwrap_or(0.0) as f32,
        Some(entries.sample),
        &mut constants,
    )?
    .0;
    let result = graph.local_for(
        entries.sample,
        "result",
        &[("M", selected), ("V", readout_vocabulary)],
    )?;
    let mut selections = Vec::new();
    let mut anchor_rows = None;
    let mut candidates = None;
    let anchors = match (&draft.method, &entries.markov) {
        (DraftMethod::DFlash, None) => {
            let mut all = result.tensor().slice_leading(0, outputs);
            let inputs =
                readout::selection_inputs(&mut graph, entries.sample, outputs, readout_vocabulary)?;
            selections.push(readout::sample(
                &mut graph,
                entries.shape,
                entries.sample,
                readout_vocabulary,
                &mut logits,
                outputs,
                class.shaped,
                inputs,
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
            let memory_dims = [
                ("M", slots),
                ("V", vocabulary),
                ("D", rank),
                ("R", direct.shape[0]),
                ("F", direct.shape[1]),
            ];
            let anchors = graph.input_for(chain.embedding, "tokens", &memory_dims)?;
            // The projection's packing bound (a bound of all its rows bounds
            // the readout vocabulary's).
            let markov_bound = PackingBound::planned(
                &mut graph,
                load,
                WeightRole {
                    scope: WeightScope::Draft,
                    kind: WeightKind::MarkovProjection,
                },
                &mut weights,
                &mut constants,
            )?;
            // An NVFP4 projection's second-level scale (absent otherwise).
            let (markov_scale, markov_scale_extent) = weight_scale_port(
                &mut graph,
                load,
                WeightRole {
                    scope: WeightScope::Draft,
                    kind: WeightKind::MarkovProjection,
                },
                &mut weights,
                &mut constants,
            )?;
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
                            directory: (&direct.tensor).into(),
                            scale: 1.0,
                            normalize: 0,
                            epsilon,
                        },
                    )?
                    .r0;
                let mut biased = graph
                    .enqueue(
                        chain.projection,
                        &[
                            ("M", outputs),
                            ("O", slots),
                            ("H", readout_vocabulary),
                            ("F", rank),
                            ("DS", markov_scale_extent),
                            ("DB", markov_bound.extent),
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
                let mut step_result = result
                    .tensor()
                    .slice_leading(step * slots, (step + 1) * slots);
                let inputs = readout::selection_inputs(
                    &mut graph,
                    entries.sample,
                    slots,
                    readout_vocabulary,
                )?;
                selections.push(readout::sample(
                    &mut graph,
                    entries.shape,
                    entries.sample,
                    readout_vocabulary,
                    &mut biased,
                    slots,
                    class.shaped,
                    inputs,
                    (&mut step_result).into(),
                )?);
                // Confidence is sigmoid(score) < 0 at the service's fixed
                // threshold. It cannot decline a selection (including a NaN
                // score), so neither the score nor its features are computed.
                constants.push(rows_of_step);
            }
            Some(anchors)
        }
        (DraftMethod::DFlash2 { selector }, None) => {
            let stages = entries
                .selector
                .as_ref()
                .ok_or("a DFlash2 draft graph has no selector entries")?;
            Some(selector_path(
                &mut graph,
                stages,
                load,
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
                    readout_vocabulary,
                    hidden,
                    rank: selector.rank,
                    top_k: selector.top_k,
                    epsilon,
                },
            )?)
        }
        (DraftMethod::LiLiCorr { .. }, None) => {
            let (corrector, stages) =
                corrector.ok_or("a LiLiCorr draft graph has no corrector entries")?;
            let (rows, selected) = correct_candidates(
                &mut graph,
                stages,
                entries.embedding,
                load,
                &mut weights,
                &mut constants,
                CorrectorPath {
                    corrector,
                    table: &mut table,
                    conditioning: &conditioning_rows,
                    entry_rows: class.entry_rows,
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
                    readout_vocabulary,
                    hidden,
                    embedding_scale: draft.embedding_scale as f32,
                    epsilon,
                },
            )?;
            graph.export(&selected)?;
            anchor_rows = Some(rows);
            candidates = Some(selected);
            None
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
            anchor_rows,
            candidates,
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

/// One `project_rows` of `rows` rows through `weight`, the planned weight of
/// `role`, binding its planned per-tensor scale and its packing `bound`.
#[allow(clippy::too_many_arguments)]
fn project_planned<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Binding<'a, project_rows::Entry>,
    load: &ModelLoadPlan,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    role: WeightRole,
    (weight, bound): (&WorkflowTensor, &PackingBound),
    source: seismic::WorkflowTensorRef<'_>,
    (rows, inputs, outputs): (u64, u64, u64),
) -> Result<WorkflowTensor, GraphError> {
    let (scale, extent) = weight_scale_port(graph, load, role, weights, constants)?;
    graph
        .enqueue(
            entry,
            &[
                ("M", rows),
                ("K", inputs),
                ("N", outputs),
                ("WS", extent),
                ("WB", bound.extent),
            ],
            project_rows::WorkflowArgs {
                source,
                weight: weight.into(),
                weight_scale: scale.tensor().into(),
                weight_bound: (&bound.tensor).into(),
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
    load: &ModelLoadPlan,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    (scope, kind): (WeightScope, WeightKind),
    source: seismic::WorkflowTensorRef<'_>,
    dims: (u64, u64, u64),
) -> Result<WorkflowTensor, GraphError> {
    let role = WeightRole { scope, kind };
    let weight_tensor = weight(graph, load, scope, kind, weights)?;
    let bound = PackingBound::planned(graph, load, role, weights, constants)?;
    project_planned(
        graph,
        entry,
        load,
        weights,
        constants,
        role,
        (&weight_tensor, &bound),
        source,
        dims,
    )
}

/// One convolved draft layer of the block pass and what it reads.
struct ConvolvedLayer<'a, 'o, G: GraphDraft + 'a> {
    entries: &'o ConvolvedLayerEntries<'a, G>,
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
    /// The layer's resolved history rows, slab rows and accepted-row window.
    history: (u64, u32),
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
    load: &ModelLoadPlan,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
) -> Result<(WorkflowTensor, LayerPorts), GraphError> {
    let ConvolvedLayer {
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
    // A projection of the block rows binds its weight's planned scale and
    // packing bound: the attention's through its loaded weights, the rest
    // loading theirs.
    let attention_scope = draft_sublayer(layer.index, 0)?;
    let project_attention = |graph: &mut G,
                             weights: &mut Vec<(WeightPort, NativePort)>,
                             constants: &mut Vec<GraphConstant>,
                             entry: G::Binding<'a, project_rows::Entry>,
                             kind: WeightKind,
                             operand: (&WorkflowTensor, &PackingBound),
                             source: seismic::WorkflowTensorRef<'_>,
                             (inputs, outputs): (u64, u64)| {
        project_planned(
            graph,
            entry,
            load,
            weights,
            constants,
            WeightRole {
                scope: attention_scope,
                kind,
            },
            operand,
            source,
            (rows, inputs, outputs),
        )
    };
    let project = |graph: &mut G,
                   weights: &mut Vec<(WeightPort, NativePort)>,
                   constants: &mut Vec<GraphConstant>,
                   entry: G::Binding<'a, project_rows::Entry>,
                   role: (WeightScope, WeightKind),
                   source: seismic::WorkflowTensorRef<'_>,
                   (inputs, outputs): (u64, u64)| {
        project_role(
            graph,
            entry,
            load,
            weights,
            constants,
            role,
            source,
            (rows, inputs, outputs),
        )
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
        let norm = weight(graph, load, scope, WeightKind::InputNorm, weights)?;
        let base = weight(graph, load, scope, WeightKind::ConvolutionBase, weights)?;
        let normed = graph
            .enqueue(
                norm_entry,
                &[
                    ("M", rows),
                    ("O", rows),
                    ("D", hidden),
                    ("PT", 1),
                    ("NM", hidden),
                ],
                readout_features_rows::WorkflowArgs {
                    hidden: (&residual.reshape(&[rows, 1, hidden])).into(),
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
        constants,
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
        return Err("a convolved draft layer projects its keys and values".into());
    };
    let query = project_attention(
        graph,
        weights,
        constants,
        entries.query,
        WeightKind::Query,
        (&attention.query, &attention.bounds.query),
        (&convolved).into(),
        (hidden, heads * width),
    )?;
    let key = project_attention(
        graph,
        weights,
        constants,
        entries.key,
        WeightKind::Key,
        (key_weight, &attention.bounds.key),
        (&convolved).into(),
        (hidden, key_width),
    )?;
    let value = project_attention(
        graph,
        weights,
        constants,
        entries.value,
        WeightKind::Value,
        (value_weight, &attention.bounds.value),
        (&convolved).into(),
        (hidden, key_width),
    )?;
    let (attended, state, controls) = attention_graph::mix(
        graph,
        layer.mix,
        (&attention.query_norm, &attention.key_norm),
        &attention.sinks,
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
            history_tiles: 0,
            shape,
            operator: layer.operator,
            epsilon,
            head_epsilon: epsilon,
            post_norm_epsilon: 0.0,
            post_norm_scale: 1.0,
            activation: layer.activation,
            inject_only: false,
            causal_cache: false,
            selection: None,
        },
        None,
        None,
    )?;
    let attended = attended.ok_or("draft block attention result is absent")?;
    let output = project_attention(
        graph,
        weights,
        constants,
        entries.output,
        WeightKind::AttentionOutput,
        (
            attention
                .output
                .as_ref()
                .ok_or("draft block output weight is absent")?,
            &attention.bounds.output,
        ),
        (&attended.reshape(&[rows, heads * width])).into(),
        (heads * width, hidden),
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
        (hidden, features),
    )?;
    let up = project(
        graph,
        weights,
        constants,
        entries.up,
        (scope, WeightKind::DenseUp),
        (&convolved).into(),
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
        weights,
        constants,
        entries.down,
        (scope, WeightKind::DenseDown),
        (&product).into(),
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
fn selector_path<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: &SelectorEntries<'a, G>,
    load: &ModelLoadPlan,
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
            &[
                ("M", path.rows),
                ("O", outputs),
                ("D", hidden),
                ("PT", 1),
                ("NM", hidden),
            ],
            readout_features_rows::WorkflowArgs {
                hidden: (&path.residual.reshape(&[path.rows, 1, hidden])).into(),
                norm: path.output_norm.into(),
                out_rows: path.head_rows.into(),
                epsilon: path.epsilon,
            },
        )?
        .value;
    let hidden_bound = PackingBound::planned(
        graph,
        load,
        WeightRole {
            scope: WeightScope::Draft,
            kind: WeightKind::SelectorHidden,
        },
        weights,
        constants,
    )?;
    let absent_scale = GraphConstant::absent(graph, constants)?.tensor().clone();
    let projected = graph
        .enqueue(
            entries.hidden,
            &[
                ("M", outputs),
                ("K", hidden),
                ("N", rank),
                ("WS", 0),
                ("WB", hidden_bound.extent),
            ],
            project_rows::WorkflowArgs {
                source: (&features).into(),
                weight: (&selector_hidden).into(),
                weight_scale: (&absent_scale).into(),
                weight_bound: (&hidden_bound.tensor).into(),
            },
        )?
        .value;
    let successors = graph
        .enqueue(
            entries.successor,
            &[
                ("M", outputs * top_k),
                ("V", vocabulary),
                ("D", rank),
                ("R", path.direct.shape[0]),
                ("F", path.direct.shape[1]),
            ],
            embedding_rows::WorkflowArgs {
                table: (&successor_codes).into(),
                tokens: (&candidates).into(),
                directory: (&path.direct.tensor).into(),
                scale: 1.0,
                normalize: 0,
                epsilon: path.epsilon,
            },
        )?
        .r0;
    let code_dims = [
        ("M", slots),
        ("V", vocabulary),
        ("D", rank),
        ("R", path.direct.shape[0]),
        ("F", path.direct.shape[1]),
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

/// The block pass's rows and tables LiLiCorr's corrector reads.
struct CorrectorPath<'t> {
    corrector: &'t CandidateCorrector,
    /// The block's token table, which the candidates' embeddings gather.
    table: &'t mut GatheredTable,
    /// The entry rows' F32 conditioning `[entry_rows, hidden]`.
    conditioning: &'t WorkflowTensor,
    entry_rows: u64,
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
    readout_vocabulary: u64,
    hidden: u64,
    embedding_scale: f32,
    epsilon: f32,
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
/// and the candidates, which the candidates' embeddings gather through the
/// table's directory: rows the device selects, which the host observes.
fn correct_candidates<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: &CorrectorEntries<'a, G>,
    embedding: G::Binding<'a, embedding_rows::Entry>,
    load: &ModelLoadPlan,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    path: CorrectorPath<'_>,
) -> Result<(NativePort, WorkflowTensor), GraphError> {
    let corrector = path.corrector;
    let (slots, mask_rows, hidden, epsilon) = (path.slots, path.mask_rows, path.hidden, path.epsilon);
    let (width, top_k, heads, factor) = (
        corrector.width,
        corrector.top_k,
        corrector.heads,
        corrector.factor,
    );
    let head_width = width / heads;
    let mlp = corrector.mlp();
    let outputs = slots * mask_rows;
    // Each block's nodes, and every node of every block.
    let nodes = mask_rows * top_k;
    let node_rows = slots * nodes;
    let draft = WeightScope::Draft;
    let vector_epsilon = corrector.vector_epsilon as f32;
    let load_weight = |graph: &mut G,
                       weights: &mut Vec<(WeightPort, NativePort)>,
                       scope: WeightScope,
                       kind: WeightKind| weight(graph, load, scope, kind, weights);
    let project = |graph: &mut G,
                   weights: &mut Vec<(WeightPort, NativePort)>,
                   constants: &mut Vec<GraphConstant>,
                   entry: G::Binding<'a, project_rows::Entry>,
                   (scope, site): (WeightScope, CorrectorSite),
                   source: seismic::WorkflowTensorRef<'_>,
                   dims: (u64, u64, u64)| {
        project_role(
            graph,
            entry,
            load,
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
                  (form, epsilon): (i32, f32),
                  (rows, columns): (u64, u64)| {
        let bias = weight(
            graph,
            load,
            scope,
            corrector_affine(site, AffinePart::Bias),
            weights,
        )?;
        let (residual, residual_rows) = match residual {
            Some(residual) => (residual.reshape(&[1, rows, columns]), 1),
            None => (
                GraphConstant::absent(graph, constants)?
                    .tensor()
                    .reshape(&[0, rows, columns]),
                0,
            ),
        };
        graph
            .enqueue(
                entry,
                &[("M", rows), ("N", columns), ("R", residual_rows)],
                corrector_rows::WorkflowArgs {
                    values: values.into(),
                    bias: (&bias).into(),
                    residual: (&residual).into(),
                    form,
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
                (count, columns): (u64, u64),
                epsilon: f32| {
        let identity = GraphConstant::identity(graph, count)?;
        let out_rows = identity.port().tensor().clone();
        constants.push(identity);
        graph
            .enqueue(
                entry,
                &[
                    ("M", count),
                    ("O", count),
                    ("D", columns),
                    ("PT", 1),
                    ("NM", columns),
                ],
                readout_features_rows::WorkflowArgs {
                    hidden: (&rows.reshape(&[count, 1, columns])).into(),
                    norm: weight.into(),
                    out_rows: (&out_rows).into(),
                    epsilon,
                },
            )
            .map(|normed| normed.value)
    };

    // Every mask row's candidates, in descending logit order.
    let top_k_dims = [
        ("M", outputs),
        ("V", path.readout_vocabulary),
        ("K", top_k),
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
        load_weight(
            graph,
            weights,
            draft,
            corrector_weight(CorrectorWeight::FeatureNorm(AffinePart::Weight)),
        )?,
        load_weight(
            graph,
            weights,
            draft,
            corrector_weight(CorrectorWeight::FeatureNorm(AffinePart::Bias)),
        )?,
    );
    let feature_up = (
        load_weight(
            graph,
            weights,
            draft,
            corrector_affine(CorrectorSite::FeatureUp, AffinePart::Weight),
        )?,
        load_weight(
            graph,
            weights,
            draft,
            corrector_affine(CorrectorSite::FeatureUp, AffinePart::Bias),
        )?,
    );
    let features = graph
        .enqueue(
            entries.features,
            &[
                ("M", outputs),
                ("V", path.readout_vocabulary),
                ("K", top_k),
                ("H", width),
            ],
            corrector_features::WorkflowArgs {
                logits: path.logits.into(),
                unary: (&unary).into(),
                norm_weight: (&feature_norm.0).into(),
                norm_bias: (&feature_norm.1).into(),
                up_weight: (&feature_up.0).into(),
                up_bias: (&feature_up.1).into(),
                epsilon: corrector.feature_norm.epsilon as f32,
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
        (node_rows, width, width),
    )?;

    // The candidates' token rows: their embeddings, projected into the
    // corrector width unless it is the draft's.
    let directory = path.table.selected(graph, load, weights, constants)?;
    let embedded = graph.enqueue(
        embedding,
        &[
            ("M", node_rows),
            ("V", path.table.rows),
            ("D", hidden),
            ("R", directory.shape[0]),
            ("F", directory.shape[1]),
        ],
        embedding_rows::WorkflowArgs {
            table: (&path.table.table).into(),
            tokens: (&candidates).into(),
            directory: (&directory.tensor).into(),
            scale: path.embedding_scale,
            normalize: 0,
            epsilon,
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
                (node_rows, hidden, width),
            )?;
            let bias = load_weight(
                graph,
                weights,
                draft,
                corrector_affine(CorrectorSite::Token, AffinePart::Bias),
            )?;
            (token, bias.reshape(&[1, width]), 1)
        }
        None => (
            embedded.r1,
            GraphConstant::absent(graph, constants)?
                .tensor()
                .reshape(&[0, width]),
            0,
        ),
    };

    // The mask rows' pass rows.
    let pass_rows = graph
        .enqueue(
            entries.pass_norm,
            &[
                ("M", path.rows),
                ("O", outputs),
                ("D", hidden),
                ("PT", 1),
                ("NM", hidden),
            ],
            readout_features_rows::WorkflowArgs {
                hidden: (&path.residual.reshape(&[path.rows, 1, hidden])).into(),
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
        (outputs, hidden, width),
    )?;

    // The nodes.
    let pass_bias = load_weight(
        graph,
        weights,
        draft,
        corrector_affine(CorrectorSite::Pass, AffinePart::Bias),
    )?;
    let feature_bias = load_weight(
        graph,
        weights,
        draft,
        corrector_affine(CorrectorSite::FeatureDown, AffinePart::Bias),
    )?;
    let slot_embedding = load_weight(
        graph,
        weights,
        draft,
        corrector_weight(CorrectorWeight::SlotEmbedding),
    )?;
    let rank_embedding = load_weight(
        graph,
        weights,
        draft,
        corrector_weight(CorrectorWeight::RankEmbedding),
    )?;
    let node_shape = [slots, mask_rows, top_k, width];
    let mut stream = graph
        .enqueue(
            entries.nodes,
            &[
                ("S", slots),
                ("J", mask_rows),
                ("K", top_k),
                ("H", width),
                ("TB", token_biases),
            ],
            corrector_nodes::WorkflowArgs {
                token: (&token.reshape(&node_shape)).into(),
                token_bias: (&token_bias).into(),
                pass: (&pass.reshape(&[slots, mask_rows, width])).into(),
                pass_bias: (&pass_bias).into(),
                feature: (&feature.reshape(&node_shape)).into(),
                feature_bias: (&feature_bias).into(),
                slot: (&slot_embedding).into(),
                rank: (&rank_embedding).into(),
            },
        )?
        .value;

    // The corrector's layers.
    let attention_bias = load_weight(
        graph,
        weights,
        draft,
        corrector_weight(CorrectorWeight::AttentionBias),
    )?;
    for (index, (layer, stages)) in corrector.layers.iter().zip(&entries.layers).enumerate() {
        let scope = WeightScope::CorrectorLayer(
            u32::try_from(index).map_err(|_| "corrector layer index exceeds u32")?,
        );
        let attention_norm = load_weight(
            graph,
            weights,
            scope,
            corrector_weight(CorrectorWeight::AttentionNorm),
        )?;
        let normed = norm(
            graph,
            constants,
            stages.attention_norm,
            &stream,
            &attention_norm,
            (node_rows, width),
            layer.attention_norm.epsilon as f32,
        )?;
        let qkv = project(
            graph,
            weights,
            constants,
            stages.attention_input,
            (scope, CorrectorSite::AttentionInput),
            (&normed).into(),
            (node_rows, width, 3 * width),
        )?;
        let input_bias = load_weight(
            graph,
            weights,
            scope,
            corrector_affine(CorrectorSite::AttentionInput, AffinePart::Bias),
        )?;
        let attended = graph
            .enqueue(
                entries.attention,
                &[
                    ("S", slots),
                    ("N", nodes),
                    ("NH", heads),
                    ("W", head_width),
                ],
                corrector_attention::WorkflowArgs {
                    qkv: (&qkv.reshape(&[slots, nodes, 3, heads, head_width])).into(),
                    input_bias: (&input_bias.reshape(&[3, heads, head_width])).into(),
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
            (&attended.reshape(&[node_rows, width])).into(),
            (node_rows, width, width),
        )?;
        stream = finish(
            graph,
            weights,
            constants,
            stages.attention_residual,
            (scope, CorrectorSite::AttentionOutput),
            &output,
            Some(&stream),
            (0, 0.0),
            (node_rows, width),
        )?;
        let mlp_norm = load_weight(
            graph,
            weights,
            scope,
            corrector_weight(CorrectorWeight::MlpNorm),
        )?;
        let normed = norm(
            graph,
            constants,
            stages.mlp_norm,
            &stream,
            &mlp_norm,
            (node_rows, width),
            layer.mlp_norm.epsilon as f32,
        )?;
        let up = project(
            graph,
            weights,
            constants,
            stages.up,
            (scope, CorrectorSite::Up),
            (&normed).into(),
            (node_rows, width, mlp),
        )?;
        let activated = finish(
            graph,
            weights,
            constants,
            stages.activated,
            (scope, CorrectorSite::Up),
            &up,
            None,
            (1, 0.0),
            (node_rows, mlp),
        )?;
        let down = project(
            graph,
            weights,
            constants,
            stages.down,
            (scope, CorrectorSite::Down),
            (&activated).into(),
            (node_rows, mlp, width),
        )?;
        stream = finish(
            graph,
            weights,
            constants,
            stages.down_residual,
            (scope, CorrectorSite::Down),
            &down,
            Some(&stream),
            (0, 0.0),
            (node_rows, width),
        )?;
    }
    let output_norm = load_weight(
        graph,
        weights,
        draft,
        corrector_weight(CorrectorWeight::OutputNorm),
    )?;
    let normed = norm(
        graph,
        constants,
        entries.output_norm,
        &stream,
        &output_norm,
        (node_rows, width),
        corrector.output_norm.epsilon as f32,
    )?;

    // Each slot's anchor state: the fused feature of its last entry row (the
    // row that selected its anchor), its context projection and the anchor
    // norm.
    let anchor_dims = [
        ("M", path.entry_rows),
        ("O", slots),
        ("D", hidden),
        ("PT", 1),
        ("NM", hidden),
    ];
    let anchor_rows = graph.input_for(entries.anchor_rows, "out_rows", &anchor_dims)?;
    let fused = graph
        .enqueue(
            entries.anchor_rows,
            &anchor_dims,
            readout_features_rows::WorkflowArgs {
                hidden: (&path
                    .conditioning
                    .reshape(&[path.entry_rows, 1, hidden]))
                    .into(),
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
        (slots, hidden, width),
    )?;
    let context = finish(
        graph,
        weights,
        constants,
        entries.context_bias,
        (draft, CorrectorSite::Context),
        &context,
        None,
        (0, 0.0),
        (slots, width),
    )?;
    let anchor_norm = load_weight(
        graph,
        weights,
        draft,
        corrector_weight(CorrectorWeight::AnchorNorm),
    )?;
    let anchor_state = norm(
        graph,
        constants,
        entries.anchor_norm,
        &context,
        &anchor_norm,
        (slots, width),
        corrector.anchor_norm.epsilon as f32,
    )?;

    // The factored vectors: each node's [x ; a ; x · a] through the factor
    // input, then the unit-length outgoing and incoming heads; the anchor
    // head of each slot's anchor state.
    let joined = graph
        .enqueue(
            entries.factors,
            &[("S", slots), ("N", nodes), ("H", width)],
            corrector_factors::WorkflowArgs {
                nodes: (&normed.reshape(&[slots, nodes, width])).into(),
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
        (&joined.reshape(&[node_rows, 3 * width])).into(),
        (node_rows, 3 * width, width),
    )?;
    let factored = finish(
        graph,
        weights,
        constants,
        entries.factor_activated,
        (draft, CorrectorSite::FactorInput),
        &factor_input,
        None,
        (1, 0.0),
        (node_rows, width),
    )?;
    let head = |graph: &mut G,
                    weights: &mut Vec<(WeightPort, NativePort)>,
                    constants: &mut Vec<GraphConstant>,
                    entry: G::Binding<'a, project_rows::Entry>,
                    site: CorrectorSite,
                    source: &WorkflowTensor,
                    count: u64| {
        let projected = project(
            graph,
            weights,
            constants,
            entry,
            (draft, site),
            source.into(),
            (count, width, factor),
        )?;
        finish(
            graph,
            weights,
            constants,
            entries.unit,
            (draft, site),
            &projected,
            None,
            (2, vector_epsilon),
            (count, factor),
        )
    };
    let outgoing = head(
        graph,
        weights,
        constants,
        entries.outgoing,
        CorrectorSite::Outgoing,
        &factored,
        node_rows,
    )?;
    let incoming = head(
        graph,
        weights,
        constants,
        entries.incoming,
        CorrectorSite::Incoming,
        &factored,
        node_rows,
    )?;
    let anchor = head(
        graph,
        weights,
        constants,
        entries.anchor,
        CorrectorSite::Anchor,
        &anchor_state,
        slots,
    )?;

    // The greedy walk of each slot's lattice.
    let factor_shape = [slots, mask_rows, top_k, factor];
    let mut selection = path.result.reshape(&[path.proposals, slots, 2]);
    graph.enqueue(
        entries.path,
        &[
            ("S", slots),
            ("J", mask_rows),
            ("K", top_k),
            ("F", factor),
            ("P", path.proposals),
        ],
        corrector_path::WorkflowArgs {
            candidates: (&candidates.reshape(&[slots, mask_rows, top_k, 2])).into(),
            incoming: (&incoming.reshape(&factor_shape)).into(),
            outgoing: (&outgoing.reshape(&factor_shape)).into(),
            anchor: (&anchor).into(),
            selection: (&mut selection).into(),
        },
    )?;
    Ok((anchor_rows, candidates))
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
    /// charged.
    pub(crate) fn prepare(
        target_device: &Device,
        draft: &AttestedDraft,
        load: &ModelLoadPlan,
        plan: &DraftProgramPlan,
        geometry: &DraftGeometry<'_>,
        classes: impl IntoIterator<Item = DraftGraphClass>,
    ) -> Result<Self, SubmitError> {
        let classes = classes.into_iter().collect::<Vec<_>>();
        let (_, layouts) =
            certify_draft_family(target_device.backend(), load, plan, geometry, &classes)
                .map_err(|error| invalid(error.to_string()))?;
        let mut prepared = BTreeMap::new();
        for class in classes {
            let class_error =
                |error: GraphError| invalid(format!("draft graph class {class:?}: {error}"));
            let parts = draft_graph(
                target_device.native_graph_with_layout(&layouts[&class]),
                DraftGraphEntries::prepared(draft),
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
        let plans = prepared
            .values()
            .map(|graph| graph.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
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
/// draft's own), the target's projection, or a drafter weight.
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
            // `ModelLoadPlan::with_readout_candidates` offers no planes of a
            // projection a separate draft reads.
            ResidentOutput::Progressive(_) => Err(invalid(
                "the draft reads an output projection the load stores as planes",
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
                    Mixer::Attention(attention) => Ok(match attention.key_value {
                        KeyValue::Owned {
                            domain: HistoryDomain::Window { tokens },
                            ..
                        } => Some(tokens),
                        _ => None,
                    }),
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
            // by the slots their rows are brought to.
            let tokens = (0..block_rows)
                .map(|row| {
                    block
                        .tokens
                        .get(row)
                        .copied()
                        .filter(|_| row < block.actual_rows)
                        .unwrap_or(0)
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
        // A corrector's candidates are table rows the device selects.
        let candidates = graph
            .block
            .as_ref()
            .and_then(|block| block.candidates.as_ref())
            .map(|candidates| (candidates, self.graphs.embedding.read_selected()));
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
        let selected_rows = candidates
            .map(|(candidates, read)| {
                owner
                    .tensor(candidates)
                    .map(|rows| SelectedRows::exported(read, rows))
                    .ok_or_else(|| invalid("draft graph omitted its candidates"))
            })
            .transpose()?;
        Ok((
            completion,
            HeadOutput {
                selections,
                selected_rows,
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
