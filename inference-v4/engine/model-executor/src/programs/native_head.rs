//! The native draft-head program. One sealed graph per head class runs a
//! head transaction in a single device submission: the entry pass over the
//! committed rows, then one chained pass per further proposal. A pass is the
//! head block (`draft_rows`, the decoder's attention and feed-forward entries,
//! the final norm) and, when drafting, the vocabulary projection and the
//! position-keyed selection. A chained pass embeds the previous pass's
//! selection (`sample_rows` result rows are `draft_rows` token rows) and
//! conditions on its output feature, so no proposal returns to the host
//! before the chain ends.

use super::{DeviceSubmission, HeadProgram};
use crate::{
    completion::CompletionWaiter,
    native::{draft_vocabulary, AttestedFeedForward, AttestedHead, AttestedHeadBlock},
    programs::{
        graph::attention::{
            self as attention_graph, AttentionBlock, AttentionGraphEntries, AttentionWeights,
            CheckedAttentionEntries,
        },
        graph::dense::{dimensions as dense_dimensions, CheckedDenseEntries, DenseGraphEntries},
        graph::draft::GraphDraft,
        graph::readout::{self, shapes, SelectionPorts},
        graph::routed::{self, CheckedRoutedEntries, RoutedGraphEntries},
        native_constants::{
            distinct_storage_bytes, CheckedGraphFamilyResources, CheckedGraphResources,
            ConstantTensors, GraphConstant,
        },
    },
    DeviceError, FeedForwardProgramSlot, GraphOutputTensor, HeadBinding, HeadLaunchCore,
    InvariantError, ModelLoadPlan, NativeGraphOutputLease, NativeGraphWorkspaceLease, ResidentHead,
    ResidentWeight, ResourceLimits, SubmitError, ValidatedHeadLaunch,
};
use magnitude_model_batching::{row_class, TargetBatchUpload, MAX_CLASS_SEGMENTS};
use magnitude_model_contracts::{
    ActivationDType, AttentionGeometry, DecoderGeometry, ExpertGeometry, WeightKind, WeightRole,
    WeightScope,
};
use magnitude_model_kernels::{
    dense_expand, dense_output, draft_rows, head_logits_rows, readout_features_rows, sample_rows,
    shape_rows,
};
use magnitude_model_state::LayerRef;
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphFamily,
    NativeGraphMetadata, NativeGraphPlan, NativePort, Tensor, WorkflowTensor,
};
use std::{collections::BTreeMap, rc::Rc};

/// Every head pass attends over up to this many visible history spans; a
/// span that is empty costs one skipped comparison.
pub(crate) const HEAD_SEGMENTS: u64 = MAX_CLASS_SEGMENTS as u64;

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
    pub steps: u64,
    pub shaped: bool,
}

/// The admitted head classes used by both prepared formation and metadata
/// assessment. A causal head has one pass; a drafting head chains up to the
/// method's proposal bound.
pub(crate) fn head_graph_classes(
    limits: ResourceLimits,
    history_rows: u64,
    proposals: usize,
) -> Result<Vec<HeadGraphClass>, String> {
    let row_classes = magnitude_model_batching::row_classes(limits.max_batch_rows)
        .into_iter()
        .map(|rows| rows as u64)
        .collect::<Vec<_>>();
    let max_rows = *row_classes
        .last()
        .ok_or_else(|| format!("batch row bound {} has no row class", limits.max_batch_rows))?;
    let slot_bound = limits.in_flight_requests.min(limits.max_batch_rows);
    let slot_classes = magnitude_model_batching::row_classes(slot_bound)
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
                steps: 0,
                shaped: false,
            });
        }
    }
    for &slots in &slot_classes {
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
                        steps,
                        shaped,
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
    logits: G::Binding<'a, head_logits_rows::Entry>,
    shape: G::Binding<'a, shape_rows::Entry>,
    sample: G::Binding<'a, sample_rows::Entry>,
}

enum HeadFeedForwardEntries<'a, G: GraphDraft + 'a> {
    Dense(DenseGraphEntries<'a, G>),
    Routed(RoutedGraphEntries<'a, G>),
}

impl<'a> HeadGraphEntries<'a, NativeGraph> {
    fn prepared(block: &'a AttestedHeadBlock, head: &'a AttestedHead) -> Self {
        Self {
            input: &block.input,
            attention: (&block.attention).into(),
            feed_forward: match &block.feed_forward {
                AttestedFeedForward::Dense(handles) => {
                    HeadFeedForwardEntries::Dense(handles.into())
                }
                AttestedFeedForward::Routed(handles) => {
                    HeadFeedForwardEntries::Routed(handles.into())
                }
            },
            features: &block.features,
            logits: &block.logits,
            shape: &head.shape,
            sample: &head.sample,
        }
    }
}

struct CheckedHeadEntries {
    input: [(&'static str, Element); 5],
    attention: CheckedAttentionEntries,
    feed_forward: CheckedHeadFeedForwardEntries,
    features: [(&'static str, Element); 2],
    logits: [(&'static str, Element); 2],
    selection: [(&'static str, Element); 0],
}

enum CheckedHeadFeedForwardEntries {
    Dense(CheckedDenseEntries),
    Routed(CheckedRoutedEntries),
}

impl CheckedHeadEntries {
    fn new(binding: HeadBinding) -> Self {
        Self {
            input: [
                ("EW", binding.embedding_table),
                ("A", binding.activation),
                ("EN", binding.embedding_norm),
                ("HN", binding.hidden_norm),
                ("CW", binding.combine),
            ],
            attention: CheckedAttentionEntries::new(crate::AttentionBinding {
                shape: binding.attention_shape,
                norm: binding.input_norm,
                query_gate: binding.query_gate,
                key: binding.key,
                value: binding.value,
                output: binding.attention_output,
                activation: binding.activation,
                history: magnitude_model_state::KvCodec::Dense,
            }),
            feed_forward: match binding.feed_forward {
                FeedForwardProgramSlot::Dense(binding) => {
                    CheckedHeadFeedForwardEntries::Dense(CheckedDenseEntries::new(binding))
                }
                FeedForwardProgramSlot::Routed(binding) => {
                    CheckedHeadFeedForwardEntries::Routed(CheckedRoutedEntries::new(binding))
                }
            },
            features: [("NW", binding.output_norm), ("A", binding.activation)],
            logits: [("OW", binding.projection), ("A", binding.activation)],
            selection: [],
        }
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
            logits: &self.logits,
            shape: &self.selection,
            sample: &self.selection,
        })
    }
}

struct HeadGraphParts<P> {
    plan: P,
    tokens: NativePort,
    conditioning: NativePort,
    out_rows: NativePort,
    passes: Vec<PassPorts>,
    constants: Vec<GraphConstant>,
    weights: Vec<(WeightRole, NativePort)>,
    /// The output weight's leading `draft_vocabulary` rows, when drafting.
    projection: Option<NativePort>,
    /// Selections `[steps * slots, 2]` when drafting; otherwise the entry
    /// pass's features, exported so the graph has a result.
    output: WorkflowTensor,
}

type PreparedHeadGraph = HeadGraphParts<NativeGraphPlan>;

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
    weights: &mut Vec<(WeightRole, NativePort)>,
) -> Result<WorkflowTensor, String> {
    let plan = load
        .weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| format!("planned head weight {role:?} is absent"))?;
    let port = graph.port(plan.resident, &plan.shape)?;
    let tensor = port.tensor().clone();
    weights.push((role, port));
    Ok(tensor)
}

fn head_graph_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: HeadGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    geometry: &DecoderGeometry,
    attention: &AttentionGeometry,
    feed_forward: FeedForwardProgramSlot,
    class: HeadGraphClass,
) -> Result<HeadGraphParts<G::Plan>, String> {
    if class.entry_rows == 0
        || class.slots == 0
        || class.slots > class.entry_rows
        || class.history_rows == 0
        || (class.steps == 0 && class.shaped)
    {
        return Err(format!("head graph class {class:?} is inconsistent"));
    }
    let hidden = geometry.hidden;
    let vocabulary = geometry.vocabulary;
    let epsilon = geometry.epsilon as f32;
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
    let head = WeightScope::HeadBlock(0);
    let table = weight!(WeightScope::Target, WeightKind::Embedding);
    let embedding_norm = weight!(head, WeightKind::HeadEmbeddingNorm);
    let hidden_norm = weight!(head, WeightKind::HeadHiddenNorm);
    let combine = weight!(head, WeightKind::HeadCombine);
    let attention_weights = AttentionWeights {
        input_norm: weight!(head, WeightKind::InputNorm),
        query_norm: weight!(head, WeightKind::QueryNorm),
        key_norm: weight!(head, WeightKind::KeyNorm),
        query_gate: weight!(head, WeightKind::QueryGate),
        key: weight!(head, WeightKind::Key),
        value: weight!(head, WeightKind::Value),
        output: weight!(head, WeightKind::AttentionOutput),
    };
    let output_norm = weight!(head, WeightKind::OutputNorm);
    let draft_vocabulary = draft_vocabulary(vocabulary);
    let projection = (class.steps > 0)
        .then(|| -> Result<_, String> {
            let plan = load
                .weights()
                .find(|plan| {
                    plan.role
                        == WeightRole {
                            scope: WeightScope::Target,
                            kind: WeightKind::Output,
                        }
                })
                .ok_or_else(|| "planned output weight is absent".to_owned())?;
            graph.port(plan.resident, &[draft_vocabulary, hidden])
        })
        .transpose()?;

    let entry_dims = [("M", class.entry_rows), ("V", vocabulary), ("D", hidden)];
    let tokens = graph.input_for(entries.input, "tokens", &entry_dims)?;
    let conditioning = graph.input_for(entries.input, "conditioning", &entry_dims)?;
    let out_rows = graph.input_for(
        entries.features,
        "out_rows",
        &[("M", class.entry_rows), ("O", class.slots), ("D", hidden)],
    )?;
    let mut selections = (class.steps > 0)
        .then(|| {
            graph.local_for(
                entries.sample,
                "result",
                &[("M", class.steps * class.slots), ("V", draft_vocabulary)],
            )
        })
        .transpose()?;
    let mut constants = Vec::new();
    let mut passes = Vec::new();
    let mut entry_features = None;
    let mut previous: Option<WorkflowTensor> = None;
    for pass in 0..class.steps.max(1) {
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
                hidden,
                segments: HEAD_SEGMENTS,
                history_rows: class.history_rows,
                heads: attention.heads,
                kv_heads: attention.kv_heads,
                width: attention.width,
                rotary: &attention.rotary,
                epsilon,
                activation: activation(geometry.activation_dtype),
            },
        )?;
        let advanced = match (&entries.feed_forward, feed_forward) {
            (HeadFeedForwardEntries::Dense(entries), FeedForwardProgramSlot::Dense(_)) => {
                let feedforward_norm = weight!(head, WeightKind::FeedForwardNorm);
                let gate_weight = weight!(head, WeightKind::DenseGate);
                let up_weight = weight!(head, WeightKind::DenseUp);
                let down_weight = weight!(head, WeightKind::DenseDown);
                let dense_rows = GraphConstant::identity(&mut graph, rows)?;
                let dense_dims = dense_dimensions(load, head, rows)?;
                let product = graph
                    .enqueue(
                        entries.expand,
                        &dense_dims,
                        dense_expand::WorkflowArgs {
                            residual: (&attended).into(),
                            norm: (&feedforward_norm).into(),
                            gate_weight: (&gate_weight).into(),
                            up_weight: (&up_weight).into(),
                            out_rows: dense_rows.port().tensor().into(),
                            eps: epsilon,
                        },
                    )?
                    .value;
                let output = graph
                    .enqueue(
                        entries.output,
                        &dense_dims,
                        dense_output::WorkflowArgs {
                            residual: (&attended).into(),
                            product: (&product).into(),
                            down_weight: (&down_weight).into(),
                            out_rows: dense_rows.port().tensor().into(),
                        },
                    )?
                    .value;
                constants.push(dense_rows);
                output
            }
            (HeadFeedForwardEntries::Routed(entries), FeedForwardProgramSlot::Routed(binding)) => {
                let shape = ExpertGeometry {
                    count: binding.experts,
                    selected: binding.selected,
                    intermediate: binding.features,
                    shared_intermediate: binding.shared,
                    normalize_selected: binding.normalize_selected,
                };
                routed::routed(
                    &mut graph,
                    RoutedGraphEntries {
                        route: entries.route,
                        expand: entries.expand,
                        output: entries.output,
                        group: entries.group,
                        experts: entries.experts,
                        combine: entries.combine,
                    },
                    load,
                    head,
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
                    norm: (&output_norm).into(),
                    out_rows: chained_rows
                        .as_ref()
                        .map_or(out_rows.tensor(), |rows| rows.port().tensor())
                        .into(),
                    epsilon,
                },
            )?
            .value;
        constants.extend(chained_rows);
        let selection = match (&projection, selections.as_mut()) {
            (Some(projection), Some(selections)) => {
                let logits = graph
                    .enqueue(
                        entries.logits,
                        &[("O", class.slots), ("V", draft_vocabulary), ("D", hidden)],
                        head_logits_rows::WorkflowArgs {
                            features: (&features).into(),
                            weight: projection.tensor().into(),
                        },
                    )?
                    .value;
                let mut result = selections
                    .tensor()
                    .slice_leading(pass * class.slots, (pass + 1) * class.slots);
                Some(readout::sample(
                    &mut graph,
                    entries.shape,
                    entries.sample,
                    draft_vocabulary,
                    (&logits).into(),
                    class.slots,
                    class.shaped,
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
    let plan = graph.seal()?;
    Ok(HeadGraphParts {
        plan,
        tokens,
        conditioning,
        out_rows,
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
    geometry: &DecoderGeometry,
    attention: &AttentionGeometry,
    binding: HeadBinding,
    classes: impl IntoIterator<Item = HeadGraphClass>,
) -> Result<CheckedGraphResources, String> {
    let checked = CheckedHeadEntries::new(binding);
    let mut family = CheckedGraphFamilyResources::new();
    for class in classes {
        let graph = head_graph_topology(
            NativeGraphMetadata::new(backend),
            checked.entries()?,
            load,
            geometry,
            attention,
            binding.feed_forward,
            class,
        )?;
        family.include(graph.plan, graph.constants);
    }
    family.finish()
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
        geometry: &DecoderGeometry,
        attention: &AttentionGeometry,
        classes: impl IntoIterator<Item = HeadGraphClass>,
    ) -> Result<Self, SubmitError> {
        let mut prepared = BTreeMap::new();
        for class in classes {
            if prepared.contains_key(&class) {
                return Err(invalid("head graph class is duplicated"));
            }
            let graph =
                Self::prepare_class(target_device, kernels, load, geometry, attention, class)?;
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
        geometry: &DecoderGeometry,
        attention: &AttentionGeometry,
        class: HeadGraphClass,
    ) -> Result<PreparedHeadGraph, SubmitError> {
        let block = kernels
            .blocks
            .first()
            .ok_or_else(|| invalid("attested head block is absent"))?;
        head_graph_topology(
            target_device.native_graph(),
            HeadGraphEntries::prepared(block, kernels),
            load,
            geometry,
            attention,
            block.binding.feed_forward,
            class,
        )
        .map_err(invalid)
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
        let output = resident.output.tensor();
        let vocabulary = output
            .extents()
            .first()
            .copied()
            .ok_or_else(|| invalid("resident output weight has no rows"))?;
        let projection = output
            .slice_leading(0, draft_vocabulary(vocabulary))
            .map_err(device)?;
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
                .map(|(role, port)| Ok((port, resident_head_weight(resident, *role)?.tensor())))
                .chain(constants.iter().map(|(port, tensor)| Ok((*port, tensor))))
                .chain(graph.projection.iter().map(|port| Ok((port, &projection))))
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
    let weight = match (role.scope, role.kind) {
        (WeightScope::Target, WeightKind::Embedding) => &resident.embedding,
        (WeightScope::Target, WeightKind::Output) => &resident.output,
        (WeightScope::HeadBlock(0), kind) => {
            let block = resident
                .blocks
                .first()
                .ok_or_else(|| invalid("resident head block is absent"))?;
            match kind {
                WeightKind::HeadEmbeddingNorm => &block.embedding_norm,
                WeightKind::HeadHiddenNorm => &block.hidden_norm,
                WeightKind::HeadCombine => &block.combine,
                WeightKind::InputNorm => &block.input_norm,
                WeightKind::QueryNorm => &block.attention.query_norm,
                WeightKind::KeyNorm => &block.attention.key_norm,
                WeightKind::QueryGate => &block.attention.query_gate,
                WeightKind::Key => &block.attention.key,
                WeightKind::Value => &block.attention.value,
                WeightKind::AttentionOutput => &block.attention.output,
                WeightKind::FeedForwardNorm => &block.feedforward_norm,
                kind @ (WeightKind::DenseGate
                | WeightKind::DenseUp
                | WeightKind::DenseDown
                | WeightKind::Router
                | WeightKind::SharedRouter
                | WeightKind::ExpertGate
                | WeightKind::ExpertUp
                | WeightKind::ExpertDown
                | WeightKind::SharedGate
                | WeightKind::SharedUp
                | WeightKind::SharedDown) => block.feedforward.weight(kind).ok_or_else(|| {
                    invalid(format!(
                        "head feed-forward role {kind:?} disagrees with its variant"
                    ))
                })?,
                WeightKind::OutputNorm => &block.output_norm,
                _ => return Err(invalid(format!("head weight role {role:?} is invalid"))),
            }
        }
        _ => return Err(invalid(format!("head weight role {role:?} is invalid"))),
    };
    Ok(weight)
}

/// One pass's attention controls padded to `rows` rows of `HEAD_SEGMENTS`
/// spans: padding rows attend to nothing and append nowhere.
struct PassControls {
    coordinates: Vec<u8>,
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
}

impl PassControls {
    fn new(pass: &TargetBatchUpload<'_>, rows: usize) -> Result<Self, SubmitError> {
        let actual = pass.actual_rows;
        if actual > rows {
            return Err(invalid("head pass has more rows than its graph class"));
        }
        let segments = HEAD_SEGMENTS as usize;
        let mut visible = vec![0_i32; rows * segments * 2];
        for (row, ranges) in pass.visible[..actual].iter().enumerate() {
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
            fresh: padded(pass.fresh, [0, 0]),
            destinations: i32_bytes((0..rows).map(|row| {
                if row < actual {
                    pass.destinations[row]
                } else {
                    -1
                }
            })),
        })
    }
}

pub struct NativeHeadProgram {
    geometry: DecoderGeometry,
    graphs: BoundHeadGraphs,
    waiter: CompletionWaiter,
}

impl NativeHeadProgram {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    pub(crate) fn new(
        geometry: DecoderGeometry,
        graphs: BoundHeadGraphs,
    ) -> Result<Self, SubmitError> {
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
        let planes = core
            .advances()
            .first()
            .ok_or_else(|| invalid("head batch has no state advance"))?
            .bindings()
            .history
            .iter()
            .filter(|plane| plane.layer == LayerRef::Head(0))
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
            steps: steps as u64,
            shaped,
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
        // Conditioning rows in entry-row order (the launch checked each
        // slot's rows against the activation width), padded with zeros.
        let row_bytes = self.geometry.hidden as usize * self.geometry.activation_dtype.bytes();
        let mut conditioning = Vec::with_capacity(entry_rows * row_bytes);
        for rows in core.conditioning() {
            conditioning.extend_from_slice(rows.bytes());
        }
        if conditioning.len() != entry.actual_rows * row_bytes {
            return Err(invalid(
                "head conditioning bytes differ from the entry rows",
            ));
        }
        conditioning.resize(entry_rows * row_bytes, 0);
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        active
            .write_input(&graph.conditioning, &conditioning)
            .map_err(device)?;
        active
            .write_input(
                &graph.tokens,
                &i32_bytes(entry.tokens.iter().flat_map(|token| [*token, 0])),
            )
            .map_err(device)?;
        active
            .write_input(
                &graph.out_rows,
                &i32_bytes((0..slots).map(|slot| entry.out_rows.get(slot).copied().unwrap_or(0))),
            )
            .map_err(device)?;
        for (index, (ports, pass)) in graph
            .passes
            .iter()
            .zip(std::iter::once(&entry).chain(&chain))
            .enumerate()
        {
            let controls = PassControls::new(pass, if index == 0 { entry_rows } else { slots })?;
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
                let words = draft_vocabulary(self.geometry.vocabulary).div_ceil(32) as usize;
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
