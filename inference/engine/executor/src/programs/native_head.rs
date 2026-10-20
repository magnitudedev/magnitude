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
use crate::operators::streams::{
    StreamsGeometry,
    graph::{CheckedStreamsEntries, StreamsEntries, mix_site},
};
use crate::operators::{
    self, CausalHeadBlock, InputForm, Join, Mixer, SublayerInput, paired_block,
};
use crate::{
    DeviceError, FeedForwardProgramSlot, GraphOutputTensor, HeadBinding, HeadLaunchCore,
    InvariantError, ModelLoadPlan, NativeGraphOutputLease, NativeGraphWorkspaceLease, ResidentHead,
    ResidentOutput, ResidentWeight, ResourceLimits, SubmitError, ValidatedHeadLaunch,
    completion::CompletionWaiter,
    native::{AttestedFeedForward, AttestedHead, AttestedHeadBlock},
    operators::attention::graph::{
        self as attention_graph, AttentionBlock, AttentionBounds, AttentionGraphEntries,
        CheckedAttentionEntries, attention_weights,
    },
    operators::dense_ffn::graph::{
        CheckedDenseEntries, DenseGraphEntries, dimensions as dense_dimensions,
    },
    operators::output::TailEntries,
    operators::routed::fused_graph::{self as routed, CheckedRoutedEntries, RoutedGraphEntries},
    programs::{
        graph::RowForm,
        graph::readout::{self, SelectionPorts, shapes},
        graph::{GraphError, draft::GraphDraft},
        native_constants::{
            CheckedGraphFamilyResources, CheckedGraphResources, ConstantTensors, GraphConstant,
            distinct_storage_bytes,
        },
        native_target_graph::{GatheredTable, PackingBound, WeightPart, WeightPort},
    },
};
use magnitude_batching::{TargetBatchUpload, row_class};
use magnitude_family_contracts::{
    ActivationDType, Decoder, ProgressivePlane, SublayerIndex, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::feature_rows;
use magnitude_kernels::{
    attention_project, dense_expand, dense_output, draft_rows, feature_copy, readout_features_rows,
    sample_rows, shape_rows,
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
    pub prefix: bool,
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
                prefix: false,
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
            prefix: false,
        });
        classes.push(HeadGraphClass {
            prefix: true,
            ..*classes.last().unwrap()
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
                        prefix: false,
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
    visible: Option<NativePort>,
    fresh: Option<NativePort>,
    destinations: NativePort,
    /// The head's history planes, in plane-descriptor order.
    planes: Vec<NativePort>,
    selection: Option<SelectionPorts>,
}

struct HeadGraphEntries<'a, G: GraphDraft + 'a> {
    streams: Option<StreamsEntries<'a, G>>,
    gather: G::Binding<'a, feature_rows::Entry>,
    input: G::Binding<'a, draft_rows::Entry>,
    copy: G::Binding<'a, feature_copy::Entry>,
    attention: AttentionGraphEntries<'a, G>,
    priming_project: G::Binding<'a, attention_project::Entry>,
    feed_forward: HeadFeedForwardEntries<'a, G>,
    features: G::Binding<'a, readout_features_rows::Entry>,
    logits: readout::ProjectionEntries<'a, G>,
    shape: G::Binding<'a, shape_rows::Entry>,
    sample: G::Binding<'a, sample_rows::Entry>,
}

enum HeadFeedForwardEntries<'a, G: GraphDraft + 'a> {
    Dense(DenseGraphEntries<'a, G>),
    Routed(RoutedGraphEntries<'a, G>),
}

impl<'a> HeadGraphEntries<'a, NativeGraph> {
    fn prepared(block: &'a AttestedHeadBlock, head: &'a AttestedHead) -> Result<Self, String> {
        Ok(Self {
            streams: block.streams.as_ref().map(Into::into),
            gather: &block.gather,
            input: &block.input,
            copy: &block.copy,
            attention: (&block.attention).into(),
            priming_project: &block.priming_project,
            feed_forward: match &block.feed_forward {
                AttestedFeedForward::Dense(handles) => {
                    HeadFeedForwardEntries::Dense(handles.into())
                }
                AttestedFeedForward::Routed(handles) => {
                    HeadFeedForwardEntries::Routed(handles.into())
                }
                // `operators::admit` keeps draft heads on the fused form.
                AttestedFeedForward::GeneralRouted(_) | AttestedFeedForward::Parallel(_) => {
                    return Err("draft head routed feed-forward form".into());
                }
            },
            features: &block.features,
            logits: readout::ProjectionEntries::prepared(&block.logits),
            shape: &head.shape,
            sample: &head.sample,
        })
    }
}

struct CheckedHeadEntries {
    streams: Option<CheckedStreamsEntries>,
    gather: [(&'static str, Element); 1],
    copy: [(&'static str, Element); 1],
    input: [(&'static str, Element); 6],
    attention: CheckedAttentionEntries,
    feed_forward: CheckedHeadFeedForwardEntries,
    features: [(&'static str, Element); 2],
    logits: readout::CheckedProjectionEntries,
    selection: [(&'static str, Element); 0],
}

enum CheckedHeadFeedForwardEntries {
    Dense(CheckedDenseEntries),
    Routed(CheckedRoutedEntries),
}

impl CheckedHeadEntries {
    fn new(binding: HeadBinding, backend: BackendName) -> Result<Self, String> {
        Ok(Self {
            streams: binding.streams.map(CheckedStreamsEntries::new),
            gather: [("A", binding.conditioning)],
            copy: [("A", binding.conditioning)],
            input: [
                ("C", binding.conditioning),
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
                    return Err("draft head routed feed-forward form".into());
                }
            },
            features: [("NW", binding.output_norm), ("A", binding.activation)],
            logits: readout::CheckedProjectionEntries::new(
                binding.projection,
                binding.output_norm,
                binding.activation,
                true,
            ),
            selection: [],
        })
    }

    fn entries(&self) -> Result<HeadGraphEntries<'_, NativeGraphMetadata>, String> {
        Ok(HeadGraphEntries {
            streams: self.streams.as_ref().map(CheckedStreamsEntries::entries),
            gather: &self.gather,
            input: &self.input,
            copy: &self.copy,
            attention: self.attention.entries()?,
            priming_project: self.attention.entries()?.project,
            feed_forward: match &self.feed_forward {
                CheckedHeadFeedForwardEntries::Dense(entries) => {
                    HeadFeedForwardEntries::Dense(entries.entries())
                }
                CheckedHeadFeedForwardEntries::Routed(entries) => {
                    HeadFeedForwardEntries::Routed(entries.entries())
                }
            },
            features: &self.features,
            logits: self.logits.entries(),
            shape: &self.selection,
            sample: &self.selection,
        })
    }
}

struct HeadGraphParts<P> {
    plan: P,
    tokens: NativePort,
    conditioning: Option<NativePort>,
    previous: Option<NativePort>,
    /// Each slot's last entry row, read by the entry pass's features; absent
    /// from a causal-only graph, which reads no features.
    out_rows: Option<NativePort>,
    passes: Vec<PassPorts>,
    constants: Vec<GraphConstant>,
    weights: Vec<(WeightPort, NativePort)>,
    /// The complete output head, when drafting.
    projection: Option<readout::ProjectionPorts>,
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
            previous: self.previous,
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
    conditioning_bytes: usize,
    classes: BTreeMap<HeadGraphClass, PreparedHeadGraph>,
    family: NativeGraphFamily,
}

pub(crate) struct BoundHeadGraphs {
    prepared: Rc<PreparedHeadGraphs>,
    bound: BTreeMap<HeadGraphClass, BoundNativeGraphPlan>,
    constants: Vec<Tensor>,
    /// The token embedding as the passes gather it: the entry pass by the
    /// host's tokens, a chained pass by the device's selections.
    embedding: crate::device_table::ResidentTable,
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

#[allow(clippy::too_many_arguments)]
fn head_graph_topology<'a, G: GraphDraft + 'a>(
    graph: G,
    entries: HeadGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: CausalHeadBlock<'_>,
    feed_forward: FeedForwardProgramSlot,
    class: HeadGraphClass,
    backend: BackendName,
) -> Result<HeadGraphParts<G::Plan>, GraphError> {
    head_graph_draft(
        graph,
        entries,
        load,
        geometry,
        head,
        feed_forward,
        class,
        backend,
    )?
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
    head: CausalHeadBlock<'_>,
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
    let streams =
        StreamsGeometry::from_residual(hidden, head.residual).map_err(|error| error.to_string())?;
    let stream_count = streams.map_or(1, |streams| streams.streams);
    let conditioning_width = hidden * stream_count;
    let residual_chain = matches!(head.chain, magnitude_family_contracts::HeadChain::Residual);
    let conditioning_element = if residual_chain {
        Element::f32()
    } else {
        activation(geometry.activation_dtype)
    };
    let form = InputForm {
        parts: stream_count,
        normalized: streams.is_none(),
    };
    let vocabulary = geometry.vocabulary;
    let paired = paired_block(head.block).map_err(|error| error.to_string())?;
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
    // The entry pass embeds the host's tokens, a chained pass the tokens
    // the device selected.
    let mut table = GatheredTable::of(
        &mut graph,
        load,
        WeightScope::Target,
        WeightKind::Embedding,
        &mut weights,
    )?;
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
    let mut constants = Vec::new();
    let bounds = AttentionBounds::planned(
        &mut graph,
        load,
        attention_scope,
        attention,
        &attention_shape,
        !inject_only,
        &mut weights,
        &mut constants,
    )?;
    // Draft head sublayers add their outputs to the residual (admission).
    let attention_weights = attention_weights(
        &attention_shape,
        attention,
        form,
        !inject_only,
        false,
        bounds,
        |kind| {
            planned_weight(
                &mut graph,
                load,
                WeightRole {
                    scope: attention_scope,
                    kind,
                },
                &mut weights,
            )
        },
    )?;
    let output_norm = if inject_only {
        None
    } else if streams.is_some() {
        Some(GraphConstant::empty(
            &mut graph,
            activation(geometry.activation_dtype),
            &mut constants,
        )?)
    } else {
        Some(weight!(head, WeightKind::OutputNorm))
    };
    let certified = readout::HeadPlans::for_recipe(load, WeightScope::Target, head, true)?
        .certifies(class.slots, backend);
    let projection = (class.steps > 0)
        .then(|| {
            readout::HeadPlans::for_recipe(load, WeightScope::Target, head, true)?
                .ports(&mut graph, certified)
        })
        .transpose()?;

    let entry_dims = [
        ("M", class.entry_rows),
        ("V", table.rows),
        ("D", hidden),
        ("S", stream_count),
        ("R", 0),
        ("F", 1),
    ];
    graph.set_class_scope(Some("head-entry"));
    let tokens = graph.input_for(entries.input, "tokens", &entry_dims)?;
    // The host writes the entry rows' conditioning, or a device-conditioned
    // class binds the leading rows of its chunk's target features.
    let conditioning = if class.prefix && class.entry_rows == 1 {
        None
    } else if class.device {
        Some(graph.port_with_class_extent(
            conditioning_element,
            &[class.entry_rows, conditioning_width],
            0,
            "head_entry_rows",
        )?)
    } else {
        Some(graph.input_for(entries.input, "conditioning", &entry_dims)?)
    };
    let prefix_conditioning = if class.prefix {
        Some(graph.port(conditioning_element, &[1, conditioning_width])?)
    } else {
        None
    };
    let assembled = if let Some(previous) = &prefix_conditioning {
        let mut rows = graph.local_for(
            entries.copy,
            "out",
            &[("M", class.entry_rows), ("D", conditioning_width)],
        )?;
        let mut first = rows.tensor_mut().slice_leading(0, 1);
        graph.set_class_scope(None);
        graph.enqueue(
            entries.copy,
            &[("M", 1), ("D", conditioning_width)],
            feature_copy::WorkflowArgs {
                input: previous.tensor().into(),
                out: (&mut first).into(),
            },
        )?;
        if class.entry_rows > 1 {
            graph.set_class_scope(Some("head-entry-tail"));
            let source = conditioning
                .as_ref()
                .unwrap()
                .tensor()
                .slice_leading(0, class.entry_rows - 1);
            let mut tail = rows.tensor_mut().slice_leading(1, class.entry_rows);
            graph.enqueue(
                entries.copy,
                &[("M", class.entry_rows - 1), ("D", conditioning_width)],
                feature_copy::WorkflowArgs {
                    input: (&source).into(),
                    out: (&mut tail).into(),
                },
            )?;
        }
        Some(rows)
    } else {
        None
    };
    graph.set_class_scope(Some("head-entry"));
    let out_rows = if inject_only {
        None
    } else {
        Some(graph.input_for(
            entries.features,
            "out_rows",
            &[
                ("M", class.entry_rows),
                ("O", class.slots),
                ("D", hidden),
                ("PT", 1),
                ("NM", hidden),
            ],
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
                let directory = table.by_rows(&mut graph, &mut constants)?;
                graph
                    .enqueue(
                        entries.input,
                        &[
                            ("M", rows),
                            ("V", table.rows),
                            ("D", hidden),
                            ("S", stream_count),
                            ("R", directory.shape[0]),
                            ("F", directory.shape[1]),
                        ],
                        draft_rows::WorkflowArgs {
                            tokens: tokens.tensor().into(),
                            directory: (&directory.tensor).into(),
                            table: (&table.table).into(),
                            conditioning: assembled
                                .as_ref()
                                .or(conditioning.as_ref())
                                .unwrap()
                                .tensor()
                                .into(),
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
                let directory = table.selected(&mut graph, load, &mut weights, &mut constants)?;
                graph
                    .enqueue(
                        entries.input,
                        &[
                            ("M", rows),
                            ("V", table.rows),
                            ("D", hidden),
                            ("S", stream_count),
                            ("R", directory.shape[0]),
                            ("F", directory.shape[1]),
                        ],
                        draft_rows::WorkflowArgs {
                            tokens: (&selected).into(),
                            directory: (&directory.tensor).into(),
                            table: (&table.table).into(),
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
        let attention_site = match (streams, entries.streams) {
            (Some(streams), Some(entries)) => Some(mix_site(
                &mut graph,
                entries,
                load,
                attention_scope,
                streams,
                &input,
                rows,
                epsilon,
                true,
                &mut weights,
            )?),
            (None, None) => None,
            _ => return Err("head streams entries disagree with its residual form".into()),
        };
        let join = Join {
            values: attention_site.as_ref().and_then(|site| site.inject.clone()),
            streams: stream_count,
            injected: streams.is_some(),
            class: (pass == 0).then_some("head_entry_rows"),
        };
        let (attended, state, controls) = attention_graph::attention(
            &mut graph,
            AttentionGraphEntries {
                project: if inject_only {
                    entries.priming_project
                } else {
                    entries.attention.project
                },
                ..entries.attention
            },
            &attention_weights,
            &mut constants,
            SublayerInput {
                rows: attention_site
                    .as_ref()
                    .map_or(&input, |site| &site.partials),
                form,
            },
            &input,
            &join,
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
                causal_cache: false,
                selection: None,
            },
        )?;
        let (Some(output_norm), Some(out_rows)) = (&output_norm, &out_rows) else {
            graph.set_class_scope(None);
            graph.export(&attended)?;
            return Ok(HeadGraphParts {
                plan: graph,
                tokens,
                conditioning,
                previous: prefix_conditioning,
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
        let feed_forward_site = match (streams, entries.streams) {
            (Some(streams), Some(entries)) => Some(mix_site(
                &mut graph,
                entries,
                load,
                feed_forward_scope,
                streams,
                &attended,
                rows,
                epsilon,
                true,
                &mut weights,
            )?),
            _ => None,
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
                let absent_scale = GraphConstant::absent(&mut graph, &mut constants)?
                    .tensor()
                    .clone();
                let mut bound = |kind| {
                    PackingBound::planned(
                        &mut graph,
                        load,
                        WeightRole {
                            scope: feed_forward_scope,
                            kind,
                        },
                        &mut weights,
                        &mut constants,
                    )
                };
                let gate_bound = bound(WeightKind::DenseGate)?;
                let up_bound = bound(WeightKind::DenseUp)?;
                let down_bound = bound(WeightKind::DenseDown)?;
                let product = graph
                    .enqueue(
                        entries.expand,
                        &[
                            dense_dims.as_slice(),
                            &[
                                ("GS", 0),
                                ("US", 0),
                                ("GB", gate_bound.extent),
                                ("UB", up_bound.extent),
                            ],
                        ]
                        .concat(),
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
                            gate_scale: (&absent_scale).into(),
                            up_scale: (&absent_scale).into(),
                            gate_bound: (&gate_bound.tensor).into(),
                            up_bound: (&up_bound.tensor).into(),
                        },
                    )?
                    .value;
                let TailEntries::Residual(dense_output) = entries.output else {
                    return Err("draft head feed-forward has a post-norm tail".into());
                };
                let output = graph
                    .enqueue(
                        dense_output,
                        &[
                            dense_dims.as_slice(),
                            &[("DS", 0), ("DB", down_bound.extent)],
                        ]
                        .concat(),
                        dense_output::WorkflowArgs {
                            residual: (&attended).into(),
                            product: (&product).into(),
                            down_weight: (&down_weight).into(),
                            out_rows: dense_rows.port().tensor().into(),
                            down_scale: (&absent_scale).into(),
                            down_bound: (&down_bound.tensor).into(),
                        },
                    )?
                    .value;
                constants.push(dense_rows);
                graph.set_class_scope((pass == 0).then_some("head-entry"));
                output
            }
            (HeadFeedForwardEntries::Routed(entries), FeedForwardProgramSlot::Routed(binding)) => {
                let shape = routed::ExpertShape::of_binding(&binding);
                let join = Join {
                    values: feed_forward_site
                        .as_ref()
                        .and_then(|site| site.inject.clone()),
                    streams: stream_count,
                    injected: streams.is_some(),
                    class: (pass == 0).then_some("head_entry_rows"),
                };
                graph.set_class_scope((pass == 0).then_some("head-entry-routed"));
                let output = routed::routed(
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
                    &mut constants,
                    SublayerInput {
                        rows: feed_forward_site
                            .as_ref()
                            .map_or(&attended, |site| &site.partials),
                        form,
                    },
                    &attended,
                    &join,
                    rows,
                    hidden,
                    &shape,
                    epsilon,
                )?;
                graph.set_class_scope((pass == 0).then_some("head-entry"));
                output
            }
            _ => return Err("head feed-forward entries and plan disagree".into()),
        };
        // The entry pass reads each slot's last entry row; a chained
        // pass has one row per slot.
        let chained_rows = (pass > 0)
            .then(|| GraphConstant::identity(&mut graph, class.slots))
            .transpose()?;
        let mixed = match (streams, entries.streams) {
            (Some(streams), Some(entries)) => {
                mix_site(
                    &mut graph,
                    entries,
                    load,
                    head,
                    streams,
                    &advanced,
                    rows,
                    epsilon,
                    false,
                    &mut weights,
                )?
                .partials
            }
            _ => advanced.clone(),
        };
        let features = graph
            .enqueue(
                entries.features,
                &[
                    ("M", rows),
                    ("O", class.slots),
                    ("D", hidden),
                    ("PT", form.parts),
                    ("NM", form.norm_extent(hidden)),
                ],
                readout_features_rows::WorkflowArgs {
                    hidden: (&mixed.reshape(&[rows, form.parts, hidden])).into(),
                    norm: output_norm.into(),
                    out_rows: chained_rows
                        .as_ref()
                        .map_or(out_rows.tensor(), |rows| rows.port().tensor())
                        .into(),
                    epsilon,
                },
            )?
            .value;
        let pass_rows = chained_rows
            .as_ref()
            .map_or(out_rows.tensor(), |rows| rows.port().tensor())
            .clone();
        let chained_features = if residual_chain {
            graph
                .enqueue(
                    entries.gather,
                    &[("M", rows), ("O", class.slots), ("D", conditioning_width)],
                    feature_rows::WorkflowArgs {
                        fused: (&advanced).into(),
                        out_rows: (&pass_rows).into(),
                    },
                )?
                .value
        } else {
            features.clone()
        };
        graph.set_class_scope(None);
        constants.extend(chained_rows);
        let selection = match (&projection, selections.as_mut()) {
            (Some(projection), Some(selections)) => {
                let projection_entries = entries.logits.borrowed();
                let input = match projection {
                    readout::ProjectionPorts::Packed { .. } => {
                        readout::ReadoutInput::PreparedFeatures {
                            features: &features,
                            canonical_rows: None,
                        }
                    }
                    readout::ProjectionPorts::Progressive { .. } => readout::ReadoutInput::RawRms {
                        hidden: &mixed,
                        norm: output_norm,
                        form,
                        out_rows: &pass_rows,
                        epsilon,
                    },
                };
                let (mut logits, inputs) = readout::project_readout(
                    &mut graph,
                    projection_entries,
                    projection,
                    input,
                    readout::ProjectionGeometry {
                        rows,
                        projected: class.slots,
                        selected: class.slots,
                        vocabulary,
                        hidden,
                        columns: vocabulary,
                    },
                    0.0,
                    Some(entries.sample),
                    &mut constants,
                )?;
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
        previous = Some(chained_features);
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
        previous: prefix_conditioning,
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
    head: CausalHeadBlock<'_>,
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
    head: CausalHeadBlock<'_>,
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
    // entry pass also distinguishes a prefix with a current-chunk tail
    // from a single previous row, independently of attention row forms.
    let mut groups: BTreeMap<
        (u64, u64, bool, bool, bool, bool, u64, u32, u64, RowForm),
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
                class.prefix,
                class.prefix && class.entry_rows > 1,
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
        let template = draft
            .plan
            .seal_template()
            .map_err(|error| GraphError::from(error).context(format!("head class {largest:?}")))?;
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
                let mut slice = NativeGraphClassSlice::new()
                    .dimension("head_entry_rows", rows)
                    .scoped("head-entry", "M", rows);
                if class.prefix && class.entry_rows > 1 {
                    slice = slice.scoped("head-entry-tail", "M", [class.entry_rows - 1]);
                }
                if class.steps == 0 {
                    return Ok::<_, String>(slice);
                }
                Ok::<_, String>(match binding.feed_forward {
                    FeedForwardProgramSlot::Dense(_) => slice
                        .scoped("head-entry-dense", "M", rows)
                        .scoped("head-entry-dense", "O", rows),
                    FeedForwardProgramSlot::GeneralRouted(_)
                    | FeedForwardProgramSlot::Parallel(_) => {
                        return Err("draft head routed feed-forward form".into());
                    }
                    FeedForwardProgramSlot::Routed(_) if routed::decodes(class.entry_rows) => {
                        slice.scoped("head-entry-routed", "M", rows)
                    }
                    FeedForwardProgramSlot::Routed(shape) => {
                        slice.scoped("head-entry-routed", "M", rows).scoped(
                            "head-entry-routed",
                            "B",
                            [routed::grouped_blocks(
                                class.entry_rows,
                                shape.experts,
                                shape.selected,
                            )?],
                        )
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layout = template
            .certify(&slices)
            .map_err(|error| GraphError::from(error).context(format!("head class {largest:?}")))?;
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
            family.include_plan(layout.plan_heap())?;
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
    head: CausalHeadBlock<'_>,
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
        head: CausalHeadBlock<'_>,
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
            conditioning_bytes: load.residual_conditioning().map_or(
                geometry.hidden as usize * geometry.activation_dtype.bytes(),
                |width| width as usize * 4,
            ),
            classes: prepared,
            family,
        })
    }

    fn prepare_class(
        target_device: &Device,
        kernels: &AttestedHead,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        head: CausalHeadBlock<'_>,
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
        let absent_scale =
            Tensor::zeros(&resident.embedding.tensor().device(), Element::f32(), &[0])
                .map_err(device)?;
        let projection = match &resident.output {
            ResidentOutput::Packed(weight) => vec![(None, weight.tensor())],
            ResidentOutput::Progressive(progressive) => ProgressivePlane::ALL
                .into_iter()
                .map(|plane| (Some(plane), progressive.plane(plane).tensor()))
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
                    Some(readout::ProjectionPorts::Packed { weight, scale, .. }) => {
                        let ResidentOutput::Packed(output) = &resident.output else {
                            return Err(invalid(
                                "the head projection and the resident output head disagree",
                            ));
                        };
                        vec![
                            projected(None).map(|tensor| (weight, tensor)),
                            Ok((scale, output.scale().unwrap_or(&absent_scale))),
                        ]
                    }
                    Some(readout::ProjectionPorts::Progressive { planes: ports, .. }) => ports
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
            embedding: resident.embedding.table().clone(),
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
    /// Visit the graph constants the program binds.
    pub(crate) fn visit_constants(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        self.graphs.constants.iter().for_each(visit);
    }

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
            device: matches!(
                core.conditioning(),
                crate::HeadConditioning::Features { .. }
            ),
            prefix: matches!(
                core.conditioning(),
                crate::HeadConditioning::Features {
                    previous: Some(_),
                    ..
                }
            ),
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
                let row_bytes = self.graphs.prepared.conditioning_bytes;
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
                    .write_input(
                        graph
                            .conditioning
                            .as_ref()
                            .ok_or_else(|| invalid("host conditioning port is absent"))?,
                        &conditioning,
                    )
                    .map_err(device)?;
            }
            // The target features' leading class rows; rows past the entry
            // rows condition padding rows, which append nowhere.
            crate::HeadConditioning::Features {
                current: features,
                previous,
            } => {
                if let (Some(port), Some(previous)) = (&graph.previous, previous) {
                    bindings
                        .set(port, previous.allocation().tensor().map_err(device)?)
                        .map_err(device)?;
                }
                if let Some(port) = &graph.conditioning {
                    let features = features
                        .tensor()
                        .slice_leading(0, entry_rows as u64)
                        .map_err(|error| invalid(format!("head conditioning features: {error}")))?;
                    bindings.set(port, &features).map_err(device)?;
                }
            }
        }
        let rows = self
            .graphs
            .embedding
            .indices(entry.tokens)
            .map_err(invalid)?;
        // A chained pass embeds the device's selection. One the device
        // does not hold embeds another row, and the target rejects the
        // proposal that follows.
        if graph.passes.len() > 1 {
            self.graphs.embedding.read_selected();
        }
        active
            .write_input(
                &graph.tokens,
                &i32_bytes(rows.iter().flat_map(|row| [*row, 0])),
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
