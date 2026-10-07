//! The native draft-head program. One sealed graph per head class runs a
//! head transaction in a single device submission: the entry pass over the
//! committed rows, then one chained pass per further proposal. A pass is the
//! head block (`draft_rows`, the decoder's attention and feed-forward entries,
//! the final norm) and, when drafting, the vocabulary projection and the
//! position-keyed selection. A chained pass embeds the previous pass's
//! selection (`sample_rows` result rows are `draft_rows` token rows) and
//! conditions on its output feature, so no proposal returns to the host
//! before the chain ends. A causal-only transaction (priming, catch-up)
//! drafts nothing: its entry pass ends once the attention has appended the
//! rows' K/V history, the only thing a later draft reads of them.

use super::{DeviceSubmission, HeadProgram};
use crate::operators::{self, paired_block, Mixer};
use crate::{
    completion::CompletionWaiter,
    native::{AttestedFeedForward, AttestedHead, AttestedHeadBlock, HeadLogitsKernels},
    operators::attention::graph::{
        self as attention_graph, attention_weights, AttentionBlock, AttentionGraphEntries,
        CheckedAttentionEntries,
    },
    operators::dense_ffn::graph::{
        dimensions as dense_dimensions, CheckedDenseEntries, DenseGraphEntries,
    },
    operators::output::TailEntries,
    operators::routed::fused_graph::{self as routed, CheckedRoutedEntries, RoutedGraphEntries},
    programs::{
        graph::readout::{self, shapes, SelectionPorts},
        graph::RowForm,
        graph::{draft::GraphDraft, GraphError},
        native_constants::{
            distinct_storage_bytes, CheckedGraphFamilyResources, CheckedGraphResources,
            ConstantTensors, GraphConstant,
        },
        native_target_graph::{WeightPart, WeightPort},
    },
    DeviceError, FeedForwardProgramSlot, GraphOutputTensor, HeadBinding, HeadLaunchCore,
    HeadProjection, InvariantError, ModelLoadPlan, NativeGraphOutputLease, NativeGraphWorkspaceLease,
    ResidentHead, ResidentOutput, ResidentWeight, ResourceLimits, SubmitError, ValidatedHeadLaunch,
};
use magnitude_batching::{row_class, TargetBatchUpload};
use magnitude_family_contracts::{
    ActivationDType, Decoder, HeadBlock, ProgressivePlane, SublayerIndex, WeightKind, WeightRole,
    WeightScope,
};
use magnitude_kernels::{
    dense_expand, dense_output, draft_rows, head_logits_rows, readout_features_rows, sample_rows,
    shape_rows,
};
use magnitude_state::LayerRef;
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphClassSlice,
    NativeGraphFamily, NativeGraphLayout, NativeGraphMetadata, NativeGraphPlan,
    NativeGraphStorageBytes, NativePort, Tensor, WorkflowTensor,
};
use std::{collections::BTreeMap, rc::Rc};

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native head program",
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

/// One head graph: the entry pass's row class, the slot class of its
/// outputs and of every chained pass, the head arena's rows, the number of
/// selections per slot (0 for a causal-only head) and whether selection is
/// shaped first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct HeadGraphClass {
    pub entry_rows: u64,
    pub slots: u64,
    pub history_rows: u64,
    pub slab_rows: u32,
    pub segments: u64,
    pub steps: u64,
    pub shaped: bool,
    /// Whether a causal-only entry of one slot reads its conditioning from a
    /// target feature output on the device (a prompt chunk's entry run
    /// behind the chunk) rather than host-written rows.
    pub device: bool,
}

/// The admitted head classes used by both prepared formation and metadata
/// assessment. A causal head has one pass, conditioned by host rows or, for
/// one slot, by the target's features on the device; a drafting head chains
/// up to the method's proposal bound.
pub(crate) fn head_graph_classes(
    limits: ResourceLimits,
    history_rows: u64,
    slab_rows: u32,
    span_limit: usize,
    proposals: usize,
) -> Result<Vec<HeadGraphClass>, String> {
    let segments = u64::try_from(
        span_limit
            .checked_next_power_of_two()
            .ok_or("head segment class overflows")?,
    )
    .map_err(|_| "head segment class exceeds u64")?;
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
    let slot_bound = limits.max_launch_slots.min(limits.max_launch_rows);
    let slot_classes = magnitude_batching::row_classes(slot_bound)
        .into_iter()
        .map(|slots| slots as u64)
        .collect::<Vec<_>>();
    let mut classes = Vec::new();
    for &entry_rows in &row_classes {
        for &slots in slot_classes.iter().filter(|slots| **slots <= entry_rows) {
            classes.push(HeadGraphClass {
                entry_rows,
                slots,
                history_rows,
                slab_rows,
                segments,
                steps: 0,
                shaped: false,
                device: false,
            });
        }
        classes.push(HeadGraphClass {
            entry_rows,
            slots: 1,
            history_rows,
            slab_rows,
            segments,
            steps: 0,
            shaped: false,
            device: true,
        });
    }
    // Drafting passes project the draft vocabulary per slot: their slots are
    // bounded by the selection bound, not the launch's request slots.
    for &slots in slot_classes
        .iter()
        .filter(|slots| **slots <= limits.max_drafting_slots as u64)
    {
        let entry_bound = row_class(
            (slots as usize)
                .saturating_mul(proposals + 1)
                .min(max_rows as usize),
        )
        .ok_or("head entry row bound has no class")? as u64;
        for &entry_rows in row_classes
            .iter()
            .filter(|rows| **rows >= slots && **rows <= entry_bound)
        {
            for steps in 1..=proposals as u64 {
                for shaped in [false, true] {
                    classes.push(HeadGraphClass {
                        entry_rows,
                        slots,
                        history_rows,
                        slab_rows,
                        segments,
                        steps,
                        shaped,
                        device: false,
                    });
                }
            }
        }
    }
    Ok(classes)
}

/// The per-run inputs of one pass.
struct PassPorts {
    coordinates: NativePort,
    visible: NativePort,
    fresh: NativePort,
    destinations: NativePort,
    /// The head's history planes, in plane-descriptor order.
    planes: Vec<NativePort>,
    selection: Option<SelectionPorts>,
}

struct HeadGraphEntries<'a, G: GraphDraft + 'a> {
    input: G::Binding<'a, draft_rows::Entry>,
    attention: AttentionGraphEntries<'a, G>,
    feed_forward: HeadFeedForwardEntries<'a, G>,
    features: G::Binding<'a, readout_features_rows::Entry>,
    logits: HeadLogitsEntries<'a, G>,
    shape: G::Binding<'a, shape_rows::Entry>,
    sample: G::Binding<'a, sample_rows::Entry>,
}

/// The head's projection entries, by its `HeadProjection`.
enum HeadLogitsEntries<'a, G: GraphDraft + 'a> {
    Packed(G::Binding<'a, head_logits_rows::Entry>),
    Progressive(readout::ProgressiveEntries<'a, G>),
}

enum HeadFeedForwardEntries<'a, G: GraphDraft + 'a> {
    Dense(DenseGraphEntries<'a, G>),
    Routed(RoutedGraphEntries<'a, G>),
}

impl<'a> HeadGraphEntries<'a, NativeGraph> {
    fn prepared(block: &'a AttestedHeadBlock, head: &'a AttestedHead) -> Result<Self, String> {
        Ok(Self {
            input: &block.input,
            attention: (&block.attention).into(),
            feed_forward: match &block.feed_forward {
                AttestedFeedForward::Dense(handles) => {
                    HeadFeedForwardEntries::Dense(handles.into())
                }
                AttestedFeedForward::Routed(handles) => {
                    HeadFeedForwardEntries::Routed(handles.into())
                }
                // `operators::admit` keeps draft heads on the fused form.
                AttestedFeedForward::GeneralRouted(_) | AttestedFeedForward::Parallel(_) => {
                    return Err("draft head routed feed-forward form".into())
                }
            },
            features: &block.features,
            logits: match &block.logits {
                HeadLogitsKernels::Packed(kernel) => HeadLogitsEntries::Packed(kernel),
                HeadLogitsKernels::Progressive(kernels) => {
                    HeadLogitsEntries::Progressive(readout::ProgressiveEntries {
                        top: &kernels.top,
                        refine: &kernels.refine,
                        exact: &kernels.exact,
                        planes: &kernels.planes,
                    })
                }
            },
            shape: &head.shape,
            sample: &head.sample,
        })
    }
}

struct CheckedHeadEntries {
    input: [(&'static str, Element); 5],
    attention: CheckedAttentionEntries,
    feed_forward: CheckedHeadFeedForwardEntries,
    features: [(&'static str, Element); 2],
    logits: CheckedHeadLogits,
    selection: [(&'static str, Element); 0],
}

/// The checked bindings of the head's projection entries.
enum CheckedHeadLogits {
    Packed([(&'static str, Element); 2]),
    /// The normed entries' (top, planes) and the later levels'.
    Progressive {
        normed: [(&'static str, Element); 2],
        level: [(&'static str, Element); 1],
    },
}

enum CheckedHeadFeedForwardEntries {
    Dense(CheckedDenseEntries),
    Routed(CheckedRoutedEntries),
}

impl CheckedHeadEntries {
    fn new(binding: HeadBinding, backend: BackendName) -> Result<Self, String> {
        Ok(Self {
            input: [
                ("EW", binding.embedding_table),
                ("A", binding.activation),
                ("EN", binding.embedding_norm),
                ("HN", binding.hidden_norm),
                ("CW", binding.combine),
            ],
            attention: CheckedAttentionEntries::new(binding.attention, false),
            feed_forward: match binding.feed_forward {
                FeedForwardProgramSlot::Dense(binding) => {
                    CheckedHeadFeedForwardEntries::Dense(CheckedDenseEntries::new(binding))
                }
                FeedForwardProgramSlot::Routed(binding) => CheckedHeadFeedForwardEntries::Routed(
                    CheckedRoutedEntries::new(binding, backend)?,
                ),
                // `operators::admit` keeps draft heads on the fused form.
                FeedForwardProgramSlot::GeneralRouted(_) | FeedForwardProgramSlot::Parallel(_) => {
                    return Err("draft head routed feed-forward form".into())
                }
            },
            features: [("NW", binding.output_norm), ("A", binding.activation)],
            logits: match binding.projection {
                HeadProjection::Packed(weight) => {
                    CheckedHeadLogits::Packed([("OW", weight), ("A", binding.activation)])
                }
                HeadProjection::Progressive => CheckedHeadLogits::Progressive {
                    normed: [("NW", binding.output_norm), ("A", binding.activation)],
                    level: [("A", binding.activation)],
                },
            },
            selection: [],
        })
    }

    fn entries(&self) -> Result<HeadGraphEntries<'_, NativeGraphMetadata>, String> {
        Ok(HeadGraphEntries {
            input: &self.input,
            attention: self.attention.entries()?,
            feed_forward: match &self.feed_forward {
                CheckedHeadFeedForwardEntries::Dense(entries) => {
                    HeadFeedForwardEntries::Dense(entries.entries())
                }
                CheckedHeadFeedForwardEntries::Routed(entries) => {
                    HeadFeedForwardEntries::Routed(entries.entries())
                }
            },
            features: &self.features,
            logits: match &self.logits {
                CheckedHeadLogits::Packed(elements) => HeadLogitsEntries::Packed(&elements[..]),
                CheckedHeadLogits::Progressive { normed, level } => {
                    HeadLogitsEntries::Progressive(readout::ProgressiveEntries {
                        top: &normed[..],
                        refine: &level[..],
                        exact: &level[..],
                        planes: &normed[..],
                    })
                }
            },
            shape: &self.selection,
            sample: &self.selection,
        })
    }
}

struct HeadGraphParts<P> {
    plan: P,
    tokens: NativePort,
    conditioning: NativePort,
    /// Each slot's last entry row, read by the entry pass's features; absent
    /// from a causal-only graph, which reads no features.
    out_rows: Option<NativePort>,
    passes: Vec<PassPorts>,
    constants: Vec<GraphConstant>,
    weights: Vec<(WeightPort, NativePort)>,
    /// The complete output head, when drafting.
    projection: Option<HeadProjectionPorts>,
    /// Selections `[steps * slots, 2]` when drafting; otherwise the entry
    /// rows' block input, exported so the graph has a result.
    output: WorkflowTensor,
}

type PreparedHeadGraph = HeadGraphParts<NativeGraphPlan>;

impl<P> HeadGraphParts<P> {
    fn map_plan<Q>(
        self,
        map: impl FnOnce(P) -> Result<Q, GraphError>,
    ) -> Result<HeadGraphParts<Q>, GraphError> {
        Ok(HeadGraphParts {
            plan: map(self.plan)?,
            tokens: self.tokens,
            conditioning: self.conditioning,
            out_rows: self.out_rows,
            passes: self.passes,
            constants: self.constants,
            weights: self.weights,
            projection: self.projection,
            output: self.output,
        })
    }
}

pub struct PreparedHeadGraphs {
    classes: BTreeMap<HeadGraphClass, PreparedHeadGraph>,
    family: NativeGraphFamily,
}

pub(crate) struct BoundHeadGraphs {
    prepared: Rc<PreparedHeadGraphs>,
    bound: BTreeMap<HeadGraphClass, BoundNativeGraphPlan>,
    constants: Vec<Tensor>,
}

fn planned_weight<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    role: WeightRole,
    weights: &mut Vec<(WeightPort, NativePort)>,
) -> Result<WorkflowTensor, GraphError> {
    let plan = load
        .weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| format!("planned head weight {role:?} is absent"))?;
    let port = graph.port(plan.resident, &plan.shape)?;
    let tensor = port.tensor().clone();
    weights.push((
        WeightPort {
            role,
            part: WeightPart::Values,
        },
        port,
    ));
    Ok(tensor)
}

/// The ports of the head's projection onto the draft vocabulary.
enum HeadProjectionPorts {
    /// The complete output projection.
    Packed(NativePort),
    /// The complete planes (the radii only when certified).
    Progressive(Vec<(ProgressivePlane, NativePort)>),
}

#[allow(clippy::too_many_arguments)]
fn head_graph_topology<'a, G: GraphDraft + 'a>(
    graph: G,
    entries: HeadGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: &HeadBlock,
    feed_forward: FeedForwardProgramSlot,
    class: HeadGraphClass,
    backend: BackendName,
) -> Result<HeadGraphParts<G::Plan>, GraphError> {
    head_graph_draft(graph, entries, load, geometry, head, feed_forward, class, backend)?
        .map_plan(|graph| graph.seal())
}

/// A head graph's passes. A progressive projection certifies its drafting
/// slots' selections up to the backend's certified bound (a proposal is only
/// ever verified, so a certified selection serves every slot it fits) and
/// projects every draft-vocabulary row beyond it.
#[allow(clippy::too_many_arguments)]
fn head_graph_draft<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: HeadGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: &HeadBlock,
    feed_forward: FeedForwardProgramSlot,
    class: HeadGraphClass,
    backend: BackendName,
) -> Result<HeadGraphParts<G>, GraphError> {
    if class.entry_rows == 0
        || class.slots == 0
        || class.slots > class.entry_rows
        || class.history_rows == 0
        || (class.steps == 0 && class.shaped)
        || (class.device && (class.steps != 0 || class.slots != 1))
    {
        return Err(format!("head graph class {class:?} is inconsistent").into());
    }
    let hidden = geometry.hidden;
    let vocabulary = geometry.vocabulary;
    let paired = paired_block(&head.block).map_err(|error| error.to_string())?;
    let Mixer::Attention(attention) = paired.mixer else {
        return Err("draft head block must attend".into());
    };
    let epsilon = paired.epsilon() as f32;
    let mut weights = Vec::new();
    macro_rules! weight {
        ($scope:expr, $kind:expr) => {
            planned_weight(
                &mut graph,
                load,
                WeightRole {
                    scope: $scope,
                    kind: $kind,
                },
                &mut weights,
            )?
        };
    }
    // The one supported head block: its own weights, then its attention and
    // feed-forward sublayers.
    let head = WeightScope::HeadBlock(0);
    let [attention_scope, feed_forward_scope] =
        [0, 1].map(|sublayer| WeightScope::HeadSublayer(SublayerIndex { block: 0, sublayer }));
    let table = weight!(WeightScope::Target, WeightKind::Embedding);
    let embedding_norm = weight!(head, WeightKind::HeadEmbeddingNorm);
    let hidden_norm = weight!(head, WeightKind::HeadHiddenNorm);
    let combine = weight!(head, WeightKind::HeadCombine);
    let attention_shape =
        operators::attention::shape(hidden, attention).map_err(|error| error.to_string())?;
    let head_epsilon = operators::attention::head_norm_epsilon(attention, paired.epsilon())
        .map_err(|error| error.to_string())? as f32;
    // A causal-only transaction only publishes its entry rows' K/V history:
    // nothing reads their attention output, so its pass ends at the append
    // and binds none of the weights after it.
    let inject_only = class.steps == 0;
    // Draft head sublayers add their outputs to the residual (admission).
    let attention_weights = attention_weights(&attention_shape, attention, !inject_only, false, |kind| {
        planned_weight(
            &mut graph,
            load,
            WeightRole {
                scope: attention_scope,
                kind,
            },
            &mut weights,
        )
    })?;
    let output_norm = if inject_only {
        None
    } else {
        Some(weight!(head, WeightKind::OutputNorm))
    };
    let certified = class.slots <= readout::certified_rows(backend);
    let projection = (class.steps > 0)
        .then(|| -> Result<_, GraphError> {
            Ok(match readout::HeadPlans::of(load)? {
                readout::HeadPlans::Packed(plan) => {
                    HeadProjectionPorts::Packed(graph.port(plan.resident, &[vocabulary, hidden])?)
                }
                readout::HeadPlans::Progressive(_) => HeadProjectionPorts::Progressive(
                    ProgressivePlane::ALL
                        .into_iter()
                        .filter(|plane| certified || *plane != ProgressivePlane::Radius)
                        .map(|plane| {
                            Ok((
                                plane,
                                graph.port(
                                    crate::progressive::element(plane),
                                    &plane.shape(vocabulary, hidden),
                                )?,
                            ))
                        })
                        .collect::<Result<_, GraphError>>()?,
                ),
            })
        })
        .transpose()?;

    let entry_dims = [("M", class.entry_rows), ("V", vocabulary), ("D", hidden)];
    graph.set_class_scope(Some("head-entry"));
    let tokens = graph.input_for(entries.input, "tokens", &entry_dims)?;
    // The host writes the entry rows' conditioning, or a device-conditioned
    // class binds the leading rows of its chunk's target features.
    let conditioning = if class.device {
        graph.port_with_class_extent(
            activation(geometry.activation_dtype),
            &[class.entry_rows, hidden],
            0,
            "head_entry_rows",
        )?
    } else {
        graph.input_for(entries.input, "conditioning", &entry_dims)?
    };
    let out_rows = if inject_only {
        None
    } else {
        Some(graph.input_for(
            entries.features,
            "out_rows",
            &[("M", class.entry_rows), ("O", class.slots), ("D", hidden)],
        )?)
    };
    graph.set_class_scope(None);
    let mut selections = (class.steps > 0)
        .then(|| {
            graph.local_for(
                entries.sample,
                "result",
                &[("M", class.steps * class.slots), ("V", vocabulary)],
            )
        })
        .transpose()?;
    let mut constants = Vec::new();
    let absent_scale = (!inject_only && matches!(feed_forward, FeedForwardProgramSlot::Dense(_)))
        .then(|| GraphConstant::absent_scale(&mut graph, &mut constants))
        .transpose()?;
    let mut passes = Vec::new();
    let mut entry_features = None;
    let mut previous: Option<WorkflowTensor> = None;
    for pass in 0..class.steps.max(1) {
        graph.set_class_scope((pass == 0).then_some("head-entry"));
        let rows = if pass == 0 {
            class.entry_rows
        } else {
            class.slots
        };
        let input = match &previous {
            None => {
                graph
                    .enqueue(
                        entries.input,
                        &[("M", rows), ("V", vocabulary), ("D", hidden)],
                        draft_rows::WorkflowArgs {
                            tokens: tokens.tensor().into(),
                            table: (&table).into(),
                            conditioning: conditioning.tensor().into(),
                            embedding_norm: (&embedding_norm).into(),
                            hidden_norm: (&hidden_norm).into(),
                            combine: (&combine).into(),
                            epsilon,
                        },
                    )?
                    .value
            }
            Some(features) => {
                let selected = selections
                    .as_ref()
                    .ok_or_else(|| "chained pass without selections".to_owned())?
                    .tensor()
                    .slice_leading((pass - 1) * class.slots, pass * class.slots);
                graph
                    .enqueue(
                        entries.input,
                        &[("M", rows), ("V", vocabulary), ("D", hidden)],
                        draft_rows::WorkflowArgs {
                            tokens: (&selected).into(),
                            table: (&table).into(),
                            conditioning: features.into(),
                            embedding_norm: (&embedding_norm).into(),
                            hidden_norm: (&hidden_norm).into(),
                            combine: (&combine).into(),
                            epsilon,
                        },
                    )?
                    .value
            }
        };
        let (attended, state, controls) = attention_graph::attention(
            &mut graph,
            entries.attention,
            &attention_weights,
            &mut constants,
            &input,
            AttentionBlock {
                rows,
                segments: class.segments,
                history_rows: class.history_rows,
                slab_rows: class.slab_rows,
                // The head's classes list no history row tiles.
                history_tiles: 0,
                shape: attention_shape,
                operator: attention,
                epsilon,
                head_epsilon,
                post_norm_epsilon: 0.0,
                post_norm_scale: 1.0,
                activation: activation(geometry.activation_dtype),
                inject_only,
            },
        )?;
        let (Some(output_norm), Some(out_rows)) = (&output_norm, &out_rows) else {
            graph.set_class_scope(None);
            graph.export(&attended)?;
            return Ok(HeadGraphParts {
                plan: graph,
                tokens,
                conditioning,
                out_rows: None,
                passes: vec![PassPorts {
                    coordinates: controls.coordinates,
                    visible: controls.visible,
                    fresh: controls.fresh,
                    destinations: controls.destinations,
                    planes: state.planes,
                    selection: None,
                }],
                constants,
                weights,
                projection: None,
                output: attended,
            });
        };
        let advanced = match (&entries.feed_forward, feed_forward) {
            (HeadFeedForwardEntries::Dense(entries), FeedForwardProgramSlot::Dense(_)) => {
                graph.set_class_scope((pass == 0).then_some("head-entry-dense"));
                let feedforward_norm = weight!(feed_forward_scope, WeightKind::InputNorm);
                let gate_weight = weight!(feed_forward_scope, WeightKind::DenseGate);
                let up_weight = weight!(feed_forward_scope, WeightKind::DenseUp);
                let down_weight = weight!(feed_forward_scope, WeightKind::DenseDown);
                let dense_rows = GraphConstant::identity_for_class(
                    &mut graph,
                    rows,
                    (pass == 0).then_some("head_entry_rows"),
                )?;
                let dense_dims = dense_dimensions(load, feed_forward_scope, rows)?;
                let Some(crate::operators::FeedForward::Dense(dense_operator)) =
                    paired.feed_forward.map(|sublayer| sublayer.op)
                else {
                    return Err("draft head dense entries without a dense feed-forward".into());
                };
                let product = graph
                    .enqueue(
                        entries.expand,
                        &[dense_dims.as_slice(), &[("GS", 0), ("US", 0)]].concat(),
                        dense_expand::WorkflowArgs {
                            residual: (&attended).into(),
                            norm: (&feedforward_norm).into(),
                            gate_weight: (&gate_weight).into(),
                            up_weight: (&up_weight).into(),
                            out_rows: dense_rows.port().tensor().into(),
                            eps: epsilon,
                            activation: operators::dense_ffn::activation_code(
                                dense_operator.up.activation(),
                            ),
                            gate_scale: absent_scale
                                .as_ref()
                                .ok_or("dense scale is absent")?
                                .into(),
                            up_scale: absent_scale.as_ref().ok_or("dense scale is absent")?.into(),
                        },
                    )?
                    .value;
                let TailEntries::Residual(dense_output) = entries.output else {
                    return Err("draft head feed-forward has a post-norm tail".into());
                };
                let output = graph
                    .enqueue(
                        dense_output,
                        &[dense_dims.as_slice(), &[("DS", 0)]].concat(),
                        dense_output::WorkflowArgs {
                            residual: (&attended).into(),
                            product: (&product).into(),
                            down_weight: (&down_weight).into(),
                            out_rows: dense_rows.port().tensor().into(),
                            down_scale: absent_scale
                                .as_ref()
                                .ok_or("dense scale is absent")?
                                .into(),
                        },
                    )?
                    .value;
                constants.push(dense_rows);
                graph.set_class_scope((pass == 0).then_some("head-entry"));
                output
            }
            (HeadFeedForwardEntries::Routed(entries), FeedForwardProgramSlot::Routed(binding)) => {
                let shape = routed::ExpertShape::of_binding(&binding);
                routed::routed(
                    &mut graph,
                    RoutedGraphEntries {
                        route: entries.route,
                        decode: entries.decode,
                        output: entries.output,
                        group: entries.group,
                        experts: entries.experts,
                        combine: entries.combine,
                    },
                    load,
                    feed_forward_scope,
                    &mut weights,
                    &attended,
                    rows,
                    hidden,
                    &shape,
                    epsilon,
                )?
            }
            _ => return Err("head feed-forward entries and plan disagree".into()),
        };
        // The entry pass reads each slot's last entry row; a chained
        // pass has one row per slot.
        let chained_rows = (pass > 0)
            .then(|| GraphConstant::identity(&mut graph, class.slots))
            .transpose()?;
        let features = graph
            .enqueue(
                entries.features,
                &[("M", rows), ("O", class.slots), ("D", hidden)],
                readout_features_rows::WorkflowArgs {
                    hidden: (&advanced).into(),
                    norm: output_norm.into(),
                    out_rows: chained_rows
                        .as_ref()
                        .map_or(out_rows.tensor(), |rows| rows.port().tensor())
                        .into(),
                    epsilon,
                },
            )?
            .value;
        graph.set_class_scope(None);
        let pass_rows = chained_rows
            .as_ref()
            .map_or(out_rows.tensor(), |rows| rows.port().tensor())
            .clone();
        constants.extend(chained_rows);
        let selection = match (&projection, selections.as_mut()) {
            (Some(projection), Some(selections)) => {
                let (mut logits, inputs) = match (projection, &entries.logits) {
                    (HeadProjectionPorts::Packed(weight), HeadLogitsEntries::Packed(entry)) => {
                        let logits = graph
                            .enqueue(
                                *entry,
                                &[("O", class.slots), ("V", vocabulary), ("D", hidden)],
                                head_logits_rows::WorkflowArgs {
                                    features: (&features).into(),
                                    weight: weight.tensor().into(),
                                },
                            )?
                            .value;
                        (logits, None)
                    }
                    (
                        HeadProjectionPorts::Progressive(ports),
                        HeadLogitsEntries::Progressive(levels),
                    ) => {
                        let plane = |plane: ProgressivePlane| {
                            ports
                                .iter()
                                .find(|(placed, _)| *placed == plane)
                                .map(|(_, port)| port.tensor())
                        };
                        let missing = || "head projection plane is absent".to_owned();
                        readout::progressive_projection(
                            &mut graph,
                            readout::ProgressiveEntries {
                                top: levels.top,
                                refine: levels.refine,
                                exact: levels.exact,
                                planes: levels.planes,
                            },
                            entries.sample,
                            readout::ProgressivePorts {
                                top: plane(ProgressivePlane::Top).ok_or_else(missing)?,
                                bit3: plane(ProgressivePlane::Bit3).ok_or_else(missing)?,
                                rest: plane(ProgressivePlane::Rest).ok_or_else(missing)?,
                                scales: plane(ProgressivePlane::Scales).ok_or_else(missing)?,
                                radius: plane(ProgressivePlane::Radius),
                            },
                            readout::ProgressiveRows {
                                hidden: &advanced,
                                norm: output_norm,
                                out_rows: &pass_rows,
                                rows,
                                projected: class.slots,
                                selected: class.slots,
                            },
                            (vocabulary, hidden),
                            epsilon,
                        )?
                    }
                    _ => return Err("head projection entries and plan disagree".into()),
                };
                let mut result = selections
                    .tensor()
                    .slice_leading(pass * class.slots, (pass + 1) * class.slots);
                let inputs = match inputs {
                    Some(inputs) => inputs,
                    None => readout::selection_inputs(
                        &mut graph,
                        entries.sample,
                        class.slots,
                        vocabulary,
                    )?,
                };
                Some(readout::sample(
                    &mut graph,
                    entries.shape,
                    entries.sample,
                    vocabulary,
                    &mut logits,
                    class.slots,
                    class.shaped,
                    inputs,
                    (&mut result).into(),
                )?)
            }
            _ => None,
        };
        passes.push(PassPorts {
            coordinates: controls.coordinates,
            visible: controls.visible,
            fresh: controls.fresh,
            destinations: controls.destinations,
            planes: state.planes,
            selection,
        });
        if pass == 0 {
            entry_features = Some(features.clone());
        }
        previous = Some(features);
    }
    let output = match &selections {
        Some(selections) => selections.tensor().clone(),
        None => entry_features.ok_or_else(|| "head graph has no entry pass".to_owned())?,
    };
    graph.export(&output)?;
    Ok(HeadGraphParts {
        plan: graph,
        tokens,
        conditioning,
        out_rows: Some(out_rows.ok_or_else(|| "head graph has no entry pass".to_owned())?),
        passes,
        constants,
        weights,
        projection,
        output,
    })
}

pub(crate) fn checked_head_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: &HeadBlock,
    binding: HeadBinding,
    classes: impl IntoIterator<Item = HeadGraphClass>,
) -> Result<CheckedGraphResources, GraphError> {
    let classes = classes.into_iter().collect::<Vec<_>>();
    certify_head_family(backend, load, geometry, head, binding, &classes)
        .map(|(resources, _)| resources)
}

fn certify_head_family(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: &HeadBlock,
    binding: HeadBinding,
    classes: &[HeadGraphClass],
) -> Result<
    (
        CheckedGraphResources,
        BTreeMap<HeadGraphClass, NativeGraphLayout>,
    ),
    GraphError,
> {
    let checked = CheckedHeadEntries::new(binding, backend)?;
    let mut family = CheckedGraphFamilyResources::new();
    let mut layouts = BTreeMap::new();
    // Every field but the entry row count fixes the graph's structure; the
    // entry pass branches only through the row form of its row count.
    let mut groups: BTreeMap<
        (u64, u64, bool, bool, u64, u32, u64, RowForm),
        Vec<HeadGraphClass>,
    > = BTreeMap::new();
    for &class in classes {
        if groups.values().any(|group| group.contains(&class)) {
            return Err("head graph class is duplicated".into());
        }
        groups
            .entry((
                class.slots,
                class.steps,
                class.shaped,
                class.device,
                class.history_rows,
                class.slab_rows,
                class.segments,
                RowForm::of(class.entry_rows),
            ))
            .or_default()
            .push(class);
    }
    for group in groups.into_values() {
        let largest = *group.iter().max_by_key(|class| class.entry_rows).unwrap();
        let draft = head_graph_draft(
            NativeGraphMetadata::new_template(backend),
            checked.entries()?,
            load,
            geometry,
            head,
            binding.feed_forward,
            largest,
            backend,
        )?;
        let template = draft.plan.seal_template()?;
        family.include(
            NativeGraphStorageBytes {
                workspace: 0,
                output: 0,
                upload: 0,
            },
            draft.constants,
        );
        let slices = group
            .iter()
            .map(|class| {
                let rows = [class.entry_rows];
                let slice = NativeGraphClassSlice::new()
                    .dimension("head_entry_rows", rows)
                    .scoped("head-entry", "M", rows);
                if class.steps == 0 {
                    return Ok::<_, String>(slice);
                }
                Ok::<_, String>(match binding.feed_forward {
                    FeedForwardProgramSlot::Dense(_) => slice
                        .scoped("head-entry-dense", "M", rows)
                        .scoped("head-entry-dense", "O", rows),
                    FeedForwardProgramSlot::GeneralRouted(_)
                    | FeedForwardProgramSlot::Parallel(_) => {
                        return Err("draft head routed feed-forward form".into())
                    }
                    FeedForwardProgramSlot::Routed(_) if routed::decodes(class.entry_rows) => slice,
                    FeedForwardProgramSlot::Routed(shape) => slice.scoped(
                        "head-entry",
                        "B",
                        [routed::grouped_blocks(
                            class.entry_rows,
                            shape.experts,
                            shape.selected,
                        )?],
                    ),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layout = template.certify(&slices)?;
        family.include(layout.storage_bytes(), []);
        for &class in &group {
            if class.steps > 0 && matches!(binding.feed_forward, FeedForwardProgramSlot::Dense(_)) {
                family.include(
                    NativeGraphStorageBytes {
                        workspace: 0,
                        output: 0,
                        upload: 0,
                    },
                    [GraphConstant::identity_value(class.entry_rows)?],
                );
            }
            layouts.insert(class, layout.clone());
        }
    }
    Ok((family.finish()?, layouts))
}

#[cfg(test)]
pub(crate) fn verify_head_family_certificates(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: &HeadBlock,
    binding: HeadBinding,
    classes: &[HeadGraphClass],
) -> Result<(), GraphError> {
    let (_, layouts) = certify_head_family(backend, load, geometry, head, binding, classes)?;
    let checked = CheckedHeadEntries::new(binding, backend)?;
    for &class in classes {
        let exact = head_graph_draft(
            NativeGraphMetadata::new(backend),
            checked.entries()?,
            load,
            geometry,
            head,
            binding.feed_forward,
            class,
            backend,
        )?;
        let charged = exact
            .plan
            .seal_with_layout(&layouts[&class])
            .map_err(|error| GraphError::from(error).context(format!("head class {class:?}")))?;
        if charged != layouts[&class].storage_bytes() {
            return Err(format!("head class {class:?} charged a different layout").into());
        }
    }
    Ok(())
}

impl PreparedHeadGraphs {
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        distinct_storage_bytes(
            self.classes
                .values()
                .flat_map(|graph| graph.constants.iter()),
        )
    }

    pub(crate) fn prepare(
        target_device: &Device,
        kernels: &AttestedHead,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        head: &HeadBlock,
        classes: impl IntoIterator<Item = HeadGraphClass>,
    ) -> Result<Self, SubmitError> {
        let classes = classes.into_iter().collect::<Vec<_>>();
        let binding = kernels
            .blocks
            .first()
            .ok_or_else(|| invalid("attested head block is absent"))?
            .binding;
        let (_, layouts) = certify_head_family(
            target_device.backend(),
            load,
            geometry,
            head,
            binding,
            &classes,
        )
        .map_err(|error| invalid(error.to_string()))?;
        let mut prepared = BTreeMap::new();
        for class in classes {
            if prepared.contains_key(&class) {
                return Err(invalid("head graph class is duplicated"));
            }
            let graph = Self::prepare_class(
                target_device,
                kernels,
                load,
                geometry,
                head,
                class,
                &layouts[&class],
            )?;
            prepared.insert(class, graph);
        }
        if prepared.is_empty() {
            return Err(invalid("head graph family has no classes"));
        }
        let plans = prepared
            .values()
            .map(|graph| graph.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self {
            classes: prepared,
            family,
        })
    }

    fn prepare_class(
        target_device: &Device,
        kernels: &AttestedHead,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        head: &HeadBlock,
        class: HeadGraphClass,
        layout: &NativeGraphLayout,
    ) -> Result<PreparedHeadGraph, SubmitError> {
        let block = kernels
            .blocks
            .first()
            .ok_or_else(|| invalid("attested head block is absent"))?;
        head_graph_topology(
            target_device.native_graph_with_layout(layout),
            HeadGraphEntries::prepared(block, kernels).map_err(invalid)?,
            load,
            geometry,
            head,
            block.binding.feed_forward,
            class,
            target_device.backend(),
        )
        .map_err(|error| invalid(error.to_string()))
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub fn workspace_bytes_max(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub fn output_bytes_max(&self) -> u64 {
        self.family.output_bytes()
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        resident: &ResidentHead,
    ) -> Result<BoundHeadGraphs, SubmitError> {
        let mut uploaded = ConstantTensors::new(resident.embedding.tensor().device());
        let projection = match &resident.output {
            ResidentOutput::Packed(weight) => vec![(None, weight.tensor())],
            ResidentOutput::Progressive(planes) => ProgressivePlane::ALL
                .into_iter()
                .map(|plane| (Some(plane), planes.plane(plane).tensor()))
                .collect(),
        };
        let projected = |plane: Option<ProgressivePlane>| -> Result<&Tensor, SubmitError> {
            projection
                .iter()
                .find(|(placed, _)| *placed == plane)
                .map(|(_, tensor)| *tensor)
                .ok_or_else(|| invalid("the head projection and the resident output head disagree"))
        };
        let mut bound = BTreeMap::new();
        for (class, graph) in &self.classes {
            let constants = graph
                .constants
                .iter()
                .map(|constant| Ok((constant.port(), uploaded.tensor(constant).map_err(device)?)))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            let fixed = graph
                .weights
                .iter()
                .map(|(weight, port)| {
                    let resident = resident_head_weight(resident, weight.role)?;
                    Ok((port, weight.part.of(resident).map_err(invalid)?))
                })
                .chain(constants.iter().map(|(port, tensor)| Ok((*port, tensor))))
                .chain(match &graph.projection {
                    None => Vec::new(),
                    Some(HeadProjectionPorts::Packed(port)) => vec![projected(None).map(|tensor| (port, tensor))],
                    Some(HeadProjectionPorts::Progressive(ports)) => ports
                        .iter()
                        .map(|(plane, port)| projected(Some(*plane)).map(|tensor| (port, tensor)))
                        .collect(),
                })
                .collect::<Result<Vec<_>, SubmitError>>()?;
            bound.insert(*class, graph.plan.bind_static(&fixed).map_err(device)?);
        }
        Ok(BoundHeadGraphs {
            prepared: self.clone(),
            bound,
            constants: uploaded.into_tensors(),
        })
    }
}

impl BoundHeadGraphs {
    /// The span class every head graph was sealed with (the head store's
    /// span limit, see [`head_graph_classes`]).
    fn segments(&self) -> Result<u64, SubmitError> {
        self.prepared
            .classes
            .keys()
            .next()
            .map(|class| class.segments)
            .ok_or_else(|| invalid("head graph family has no classes"))
    }

    fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.constants.iter().try_fold(0u64, |bytes, tensor| {
            bytes
                .checked_add(tensor.storage_bytes())
                .ok_or("head graph constant charge overflows")
        })
    }

    fn class(
        &self,
        class: HeadGraphClass,
    ) -> Result<(&PreparedHeadGraph, &BoundNativeGraphPlan), SubmitError> {
        let graph = self
            .prepared
            .classes
            .get(&class)
            .ok_or_else(|| invalid(format!("head graph class {class:?} was not prepared")))?;
        let bound = self
            .bound
            .get(&class)
            .ok_or_else(|| invalid(format!("head graph class {class:?} was not bound")))?;
        Ok((graph, bound))
    }
}

fn resident_head_weight(
    resident: &ResidentHead,
    role: WeightRole,
) -> Result<&ResidentWeight, SubmitError> {
    match (role.scope, role.kind) {
        (WeightScope::Target, WeightKind::Embedding) => Ok(&resident.embedding),
        _ => resident.weights.get(role).map_err(invalid),
    }
}

/// One pass's attention controls padded to `rows` rows of the graph's span class
/// spans: padding rows attend to nothing and append nowhere.
struct PassControls {
    coordinates: Vec<u8>,
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
}

impl PassControls {
    fn new(
        pass: &TargetBatchUpload<'_>,
        rows: usize,
        segments: usize,
    ) -> Result<Self, SubmitError> {
        let actual = pass.actual_rows;
        if actual > rows {
            return Err(invalid("head pass has more rows than its graph class"));
        }
        // The draft head's store has one Token history domain.
        let [history] = pass.histories else {
            return Err(invalid("head pass must carry exactly one history domain"));
        };
        let mut visible = vec![0_i32; rows * segments * 2];
        for (row, ranges) in history.visible[..actual].iter().enumerate() {
            let used = ranges
                .iter()
                .rposition(|range| range[1] > range[0])
                .map_or(0, |last| last + 1);
            if used > segments {
                return Err(invalid(format!(
                    "head row attends {used} history spans; the head admits {segments}"
                )));
            }
            for (span, [start, end]) in ranges[..used].iter().enumerate() {
                let at = (row * segments + span) * 2;
                visible[at] = *start;
                visible[at + 1] = *end;
            }
        }
        let padded = |values: &[[i32; 2]], fill: [i32; 2]| {
            i32_bytes((0..rows).flat_map(|row| {
                values
                    .get(row)
                    .filter(|_| row < actual)
                    .copied()
                    .unwrap_or(fill)
            }))
        };
        Ok(Self {
            coordinates: i32_bytes((0..rows).flat_map(|row| {
                pass.coordinates
                    .get(row)
                    .filter(|_| row < actual)
                    .copied()
                    .unwrap_or([0; 4])
            })),
            visible: i32_bytes(visible),
            fresh: padded(&history.fresh, [0, 0]),
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

pub struct NativeHeadProgram {
    geometry: Decoder,
    graphs: BoundHeadGraphs,
    waiter: CompletionWaiter,
}

impl NativeHeadProgram {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    pub(crate) fn new(geometry: Decoder, graphs: BoundHeadGraphs) -> Result<Self, SubmitError> {
        let waiter = CompletionWaiter::spawn().map_err(device)?;
        Ok(Self {
            geometry,
            graphs,
            waiter,
        })
    }

    /// Queue one head transaction. Returns its completion and, when
    /// drafting, the selections the graph fills.
    fn queue(
        &self,
        core: &HeadLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: NativeGraphOutputLease,
    ) -> Result<(seismic::NativeGraphCompletion, Option<GraphOutputTensor>), SubmitError> {
        let batch = core.batch();
        let entry = batch.upload();
        let chain = batch.chain().collect::<Vec<_>>();
        let steps = batch.steps();
        let actual_slots = batch.actual_slots();
        let slots = row_class(actual_slots).ok_or_else(|| invalid("head slots have no class"))?;
        let entry_rows = entry.class.rows();
        let shaped = std::iter::once(&entry)
            .chain(&chain)
            .any(|pass| pass.shaping[..pass.select_rows.len()].iter().any(shapes));
        let history = core
            .advances()
            .first()
            .ok_or_else(|| invalid("head batch has no state advance"))?
            .bindings()
            .history
            .iter()
            .filter(|plane| plane.layer == LayerRef::Head(0))
            .cloned()
            .collect::<Vec<_>>();
        let slab_rows = history
            .first()
            .ok_or_else(|| invalid("head history has no plane"))?
            .slab_rows;
        let planes = history
            .iter()
            .map(|plane| plane.buffer.clone())
            .collect::<Vec<Tensor>>();
        let history_rows = planes
            .first()
            .and_then(|plane| plane.extents().first().copied())
            .ok_or_else(|| invalid("head history has no plane"))?;
        let class = HeadGraphClass {
            entry_rows: entry_rows as u64,
            slots: slots as u64,
            history_rows,
            slab_rows,
            segments: self.graphs.segments()?,
            steps: steps as u64,
            shaped,
            device: matches!(core.conditioning(), crate::HeadConditioning::Features(_)),
        };
        let (graph, bound) = self.graphs.class(class)?;
        if graph.passes.len() != chain.len() + 1 {
            return Err(invalid("head graph passes differ from the batch's steps"));
        }
        let mut bindings = bound.bindings();
        for pass in &graph.passes {
            if pass.planes.len() != planes.len() {
                return Err(invalid(
                    "head history planes differ from the attention entry",
                ));
            }
            for (port, plane) in pass.planes.iter().zip(&planes) {
                bindings.set(port, plane).map_err(device)?;
            }
        }
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        match core.conditioning() {
            // Host rows in entry-row order (the launch checked each slot's
            // rows against the activation width), padded with zeros.
            crate::HeadConditioning::Rows(rows) => {
                let row_bytes =
                    self.geometry.hidden as usize * self.geometry.activation_dtype.bytes();
                let mut conditioning = Vec::with_capacity(entry_rows * row_bytes);
                for rows in rows {
                    conditioning.extend_from_slice(rows.bytes());
                }
                if conditioning.len() != entry.actual_rows * row_bytes {
                    return Err(invalid(
                        "head conditioning bytes differ from the entry rows",
                    ));
                }
                conditioning.resize(entry_rows * row_bytes, 0);
                active
                    .write_input(&graph.conditioning, &conditioning)
                    .map_err(device)?;
            }
            // The target features' leading class rows; rows past the entry
            // rows condition padding rows, which append nowhere.
            crate::HeadConditioning::Features(features) => {
                let features = features
                    .tensor()
                    .slice_leading(0, entry_rows as u64)
                    .map_err(|error| invalid(format!("head conditioning features: {error}")))?;
                bindings.set(&graph.conditioning, &features).map_err(device)?;
            }
        }
        active
            .write_input(
                &graph.tokens,
                &i32_bytes(entry.tokens.iter().flat_map(|token| [*token, 0])),
            )
            .map_err(device)?;
        if let Some(out_rows) = &graph.out_rows {
            active
                .write_input(
                    out_rows,
                    &i32_bytes(
                        (0..slots).map(|slot| entry.out_rows.get(slot).copied().unwrap_or(0)),
                    ),
                )
                .map_err(device)?;
        }
        for (index, (ports, pass)) in graph
            .passes
            .iter()
            .zip(std::iter::once(&entry).chain(&chain))
            .enumerate()
        {
            let controls = PassControls::new(
                pass,
                if index == 0 { entry_rows } else { slots },
                class.segments as usize,
            )?;
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
                .map_err(device)?;
            if let Some(selection) = &ports.selection {
                let words = self.geometry.vocabulary.div_ceil(32) as usize;
                readout::write_selection(pass, &mut active, selection, slots, words)?;
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
                    .ok_or_else(|| invalid("head graph omitted its selections"))
            })
            .transpose()?;
        Ok((completion, selections))
    }
}

impl HeadProgram for NativeHeadProgram {
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
                None => Err(invalid("head graph output lease is absent")),
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
