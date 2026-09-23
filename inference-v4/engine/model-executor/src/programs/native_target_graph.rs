//! Seismic-owned decoder block graphs. The engine supplies semantic model
//! topology and external state; checked entry contracts derive every stage
//! edge and intermediate allocation.

use crate::{
    ModelLoadPlan, ResidentBlockWeights, ResidentFeedForwardWeights, ResidentMixerWeights,
    ResidentTarget, ResidentWeight, ResourceLimits, StateResourcePlan,
    native::{
        AttentionKernels, AttestedFeedForward, AttestedMixer, AttestedState, AttestedTarget,
        AttestedTargetBlock, DenseKernels, RecurrentKernels, RoutedKernels,
    },
};
use magnitude_model_batching::MAX_CLASS_SEGMENTS;
use magnitude_model_contracts::{
    DecoderGeometry, FeedForwardGeometry, MixerGeometry, RecurrentHeadMapping, RotarySemantics,
    WeightKind, WeightRole, WeightScope,
};
use magnitude_model_kernels::{
    copy_rows, qwen_attention_attend, qwen_attention_normalize, qwen_attention_output,
    qwen_attention_prepare, qwen_attention_project, qwen_dense_expand, qwen_dense_output,
    qwen_embedding_rows, qwen_recurrent_mix, qwen_recurrent_normalize, qwen_recurrent_output,
    qwen_recurrent_prepare, qwen_recurrent_project, qwen_recurrent_scan, qwen_routed_expand,
    qwen_routed_logits, qwen_routed_normalize, qwen_routed_output, qwen_routed_select,
};
use seismic::{
    BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphFamily,
    NativeGraphFamilyOutputSlot, NativeGraphFamilySlot, NativeGraphPlan, NativePort,
    WorkflowTensor,
};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone)]
pub struct PreparedTargetGraphs {
    entries: BTreeMap<u64, PreparedTargetEntryGraph>,
    classes: BTreeMap<(u64, u64, u64), Vec<PreparedTargetBlockGraph>>,
    family: NativeGraphFamily,
    max_output_bytes: u64,
}

impl PreparedTargetGraphs {
    pub(crate) fn prepare(
        device: &Device,
        handles: &AttestedTarget,
        state_copies: &AttestedState,
        load: &ModelLoadPlan,
        geometry: &DecoderGeometry,
        state: &StateResourcePlan,
        limits: ResourceLimits,
    ) -> Result<Self, String> {
        let max_rows = u64::try_from(limits.max_batch_rows)
            .map_err(|_| "target batch row bound exceeds u64")?
            .checked_next_power_of_two()
            .ok_or("target batch row class overflows")?;
        let max_slots = u64::try_from(limits.in_flight_requests)
            .map_err(|_| "target request slot bound exceeds u64")?;
        let history_rows = u64::try_from(state.target_state().history_rows)
            .map_err(|_| "target history row bound exceeds u64")?;
        let mut classes = BTreeMap::new();
        let mut entries = BTreeMap::new();
        let mut plans = Vec::new();
        let mut max_output_bytes = 0u64;
        let trace = std::env::var_os("MAGNITUDE_V4_TRACE_GRAPH_SEAL").is_some();
        let mut rows = 1u64;
        while rows <= max_rows {
            let entry = PreparedTargetEntryGraph::prepare(
                device,
                &handles.embedding,
                load,
                geometry,
                rows,
            )?;
            max_output_bytes = max_output_bytes.max(entry.plan.output_bytes());
            plans.push(entry.plan.clone());
            entries.insert(rows, entry);
            let mut segments = 1u64;
            while segments <= MAX_CLASS_SEGMENTS as u64 {
                for slots in 1..=max_slots {
                    let began = Instant::now();
                    let mut blocks = Vec::with_capacity(handles.blocks.len());
                    for (index, handle) in handles.blocks.iter().enumerate() {
                        let block = PreparedTargetBlockGraph::prepare(device, handle, state_copies,
                            load, geometry, state, index, rows, segments, slots, history_rows)
                            .map_err(|error| format!(
                                "target graph row class {rows}, history segments {segments}, request slots {slots}, block {index}: {error}"
                            ))?;
                        max_output_bytes = max_output_bytes.max(block.plan.output_bytes());
                        plans.push(block.plan.clone());
                        blocks.push(block);
                    }
                    classes.insert((rows, segments, slots), blocks);
                    if trace {
                        eprintln!(
                            "target graph sealed rows={rows} segments={segments} slots={slots} blocks={} elapsed_ms={:.3}",
                            handles.blocks.len(),
                            began.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
                segments *= 2;
            }
            rows *= 2;
        }
        let family = NativeGraphFamily::new(&plans).map_err(|error| error.to_string())?;
        Ok(Self {
            entries,
            classes,
            family,
            max_output_bytes,
        })
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub fn output_bytes(&self) -> u64 {
        self.max_output_bytes
    }

    pub fn new_slot(&self) -> Result<NativeGraphFamilySlot, seismic::TensorError> {
        self.family.new_slot()
    }

    pub fn new_output_slot(&self) -> Result<NativeGraphFamilyOutputSlot, seismic::TensorError> {
        self.family.new_output_slot()
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub(crate) fn bind_weights(
        &self,
        resident: &ResidentTarget,
    ) -> Result<BoundTargetGraphs, String> {
        let mut entry_bound = BTreeMap::new();
        for (rows, entry) in &self.entries {
            let fixed = [(&entry.table, resident.embedding.tensor())];
            entry_bound.insert(
                *rows,
                entry
                    .plan
                    .bind_static(&fixed)
                    .map_err(|error| format!("target embedding graph class {rows}: {error}"))?,
            );
        }
        let mut bound = BTreeMap::new();
        for (class, blocks) in &self.classes {
            let mut class_bound = Vec::with_capacity(blocks.len());
            for (index, graph) in blocks.iter().enumerate() {
                let block = resident
                    .blocks
                    .get(index)
                    .ok_or("resident target block missing")?;
                let fixed = graph
                    .weights
                    .iter()
                    .map(|(role, port)| {
                        let weight = block_weight(block, role.kind)?;
                        Ok((port, weight.tensor()))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                class_bound.push(graph.plan.bind_static(&fixed).map_err(|error| {
                    format!("target graph static binding class {class:?} block {index}: {error}")
                })?);
            }
            bound.insert(*class, class_bound);
        }
        Ok(BoundTargetGraphs {
            prepared: self.clone(),
            entry_bound,
            bound,
        })
    }
}

pub(crate) struct BoundTargetGraphs {
    pub prepared: PreparedTargetGraphs,
    pub entry_bound: BTreeMap<u64, BoundNativeGraphPlan>,
    pub bound: BTreeMap<(u64, u64, u64), Vec<BoundNativeGraphPlan>>,
}

impl BoundTargetGraphs {
    pub(crate) fn entry(
        &self,
        rows: u64,
    ) -> Result<(&PreparedTargetEntryGraph, &BoundNativeGraphPlan), String> {
        let graph = self
            .prepared
            .entries
            .get(&rows)
            .ok_or_else(|| format!("target embedding class {rows} was not sealed"))?;
        let bound = self
            .entry_bound
            .get(&rows)
            .ok_or_else(|| format!("target embedding class {rows} was not bound"))?;
        Ok((graph, bound))
    }
    pub(crate) fn block(
        &self,
        rows: u64,
        segments: u64,
        slots: u64,
        index: usize,
    ) -> Result<(&PreparedTargetBlockGraph, &BoundNativeGraphPlan), String> {
        let key = (rows, segments, slots);
        let graph = self
            .prepared
            .classes
            .get(&key)
            .and_then(|blocks| blocks.get(index))
            .ok_or_else(|| format!("target graph class {key:?} block {index} was not sealed"))?;
        let bound = self
            .bound
            .get(&key)
            .and_then(|blocks| blocks.get(index))
            .ok_or_else(|| format!("target graph class {key:?} block {index} was not bound"))?;
        Ok((graph, bound))
    }
}

#[derive(Clone)]
pub(crate) struct PreparedTargetEntryGraph {
    pub plan: NativeGraphPlan,
    pub table: NativePort,
    pub tokens: NativePort,
    pub hidden: WorkflowTensor,
}

impl PreparedTargetEntryGraph {
    fn prepare(
        device: &Device,
        embedding: &seismic::NativeKernel<qwen_embedding_rows::Entry>,
        load: &ModelLoadPlan,
        geometry: &DecoderGeometry,
        rows: u64,
    ) -> Result<Self, String> {
        let mut graph = device.native_graph();
        let mut weights = Vec::new();
        let table_tensor = weight(
            &mut graph,
            load,
            WeightScope::Target,
            WeightKind::Embedding,
            &mut weights,
        )?;
        let table = weights
            .pop()
            .ok_or("target embedding weight port is absent")?
            .1;
        let tokens = graph
            .input_for(
                embedding,
                "tokens",
                &[
                    ("M", rows),
                    ("V", geometry.vocabulary),
                    ("D", geometry.hidden),
                ],
            )
            .map_err(|error| error.to_string())?;
        let result = graph
            .enqueue(
                embedding,
                qwen_embedding_rows::WorkflowArgs {
                    table: (&table_tensor).into(),
                    tokens: tokens.tensor().into(),
                },
            )
            .map_err(|error| error.to_string())?;
        graph
            .export(&result.r1)
            .map_err(|error| error.to_string())?;
        let plan = graph.seal().map_err(|error| error.to_string())?;
        Ok(Self {
            plan,
            table,
            tokens,
            hidden: result.r1,
        })
    }
}

fn block_weight(block: &ResidentBlockWeights, kind: WeightKind) -> Result<&ResidentWeight, String> {
    let weight = match kind {
        WeightKind::InputNorm => &block.input_norm,
        WeightKind::FeedForwardNorm => &block.feedforward_norm,
        WeightKind::QueryGate => match &block.mixer {
            ResidentMixerWeights::Attention(weights) => &weights.query_gate,
            _ => return Err("attention weight on recurrent block".into()),
        },
        WeightKind::Key => match &block.mixer {
            ResidentMixerWeights::Attention(weights) => &weights.key,
            _ => return Err("attention weight on recurrent block".into()),
        },
        WeightKind::Value => match &block.mixer {
            ResidentMixerWeights::Attention(weights) => &weights.value,
            _ => return Err("attention weight on recurrent block".into()),
        },
        WeightKind::QueryNorm => match &block.mixer {
            ResidentMixerWeights::Attention(weights) => &weights.query_norm,
            _ => return Err("attention weight on recurrent block".into()),
        },
        WeightKind::KeyNorm => match &block.mixer {
            ResidentMixerWeights::Attention(weights) => &weights.key_norm,
            _ => return Err("attention weight on recurrent block".into()),
        },
        WeightKind::AttentionOutput => match &block.mixer {
            ResidentMixerWeights::Attention(weights) => &weights.output,
            _ => return Err("attention weight on recurrent block".into()),
        },
        WeightKind::RecurrentQueryKeyValue => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.query_key_value,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::RecurrentGate => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.gate,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::RecurrentAlpha => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.alpha,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::RecurrentBeta => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.beta,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::RecurrentConvolution => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.convolution,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::RecurrentDecay => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.decay,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::RecurrentTimeBias => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.time_bias,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::RecurrentNorm => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.norm,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::RecurrentOutput => match &block.mixer {
            ResidentMixerWeights::Recurrent(weights) => &weights.output,
            _ => return Err("recurrent weight on attention block".into()),
        },
        WeightKind::DenseGate => match &block.feedforward {
            ResidentFeedForwardWeights::Dense(weights) => &weights.gate,
            _ => return Err("dense weight on routed block".into()),
        },
        WeightKind::DenseUp => match &block.feedforward {
            ResidentFeedForwardWeights::Dense(weights) => &weights.up,
            _ => return Err("dense weight on routed block".into()),
        },
        WeightKind::DenseDown => match &block.feedforward {
            ResidentFeedForwardWeights::Dense(weights) => &weights.down,
            _ => return Err("dense weight on routed block".into()),
        },
        WeightKind::Router => match &block.feedforward {
            ResidentFeedForwardWeights::Routed(weights) => &weights.router,
            _ => return Err("routed weight on dense block".into()),
        },
        WeightKind::SharedRouter => match &block.feedforward {
            ResidentFeedForwardWeights::Routed(weights) => &weights.shared_router,
            _ => return Err("routed weight on dense block".into()),
        },
        WeightKind::ExpertGate => match &block.feedforward {
            ResidentFeedForwardWeights::Routed(weights) => &weights.expert_gate,
            _ => return Err("routed weight on dense block".into()),
        },
        WeightKind::ExpertUp => match &block.feedforward {
            ResidentFeedForwardWeights::Routed(weights) => &weights.expert_up,
            _ => return Err("routed weight on dense block".into()),
        },
        WeightKind::ExpertDown => match &block.feedforward {
            ResidentFeedForwardWeights::Routed(weights) => &weights.expert_down,
            _ => return Err("routed weight on dense block".into()),
        },
        WeightKind::SharedGate => match &block.feedforward {
            ResidentFeedForwardWeights::Routed(weights) => &weights.shared_gate,
            _ => return Err("routed weight on dense block".into()),
        },
        WeightKind::SharedUp => match &block.feedforward {
            ResidentFeedForwardWeights::Routed(weights) => &weights.shared_up,
            _ => return Err("routed weight on dense block".into()),
        },
        WeightKind::SharedDown => match &block.feedforward {
            ResidentFeedForwardWeights::Routed(weights) => &weights.shared_down,
            _ => return Err("routed weight on dense block".into()),
        },
        _ => {
            return Err(format!(
                "weight kind {kind:?} does not belong to a decoder block"
            ));
        }
    };
    Ok(weight)
}

#[derive(Clone)]
pub(crate) struct PreparedTargetBlockGraph {
    pub plan: NativeGraphPlan,
    pub hidden: NativePort,
    pub weights: Vec<(WeightRole, NativePort)>,
    pub state: BlockStatePorts,
    pub controls: BlockControlPorts,
    pub routed_rows: Option<NativePort>,
    pub output: WorkflowTensor,
}

#[derive(Clone)]
pub(crate) enum BlockStatePorts {
    Attention { key: NativePort, value: NativePort },
    Recurrent { slots: Vec<RecurrentSlotPorts> },
}

#[derive(Clone)]
pub(crate) struct RecurrentSlotPorts {
    pub previous_window: NativePort,
    pub previous_delta: NativePort,
    pub following_window: NativePort,
    pub following_delta: NativePort,
    pub window_from: NativePort,
    pub window_to: NativePort,
    pub delta_from: NativePort,
    pub delta_to: NativePort,
    pub window_scatter_from: NativePort,
    pub window_scatter_to: NativePort,
    pub delta_scatter_from: NativePort,
    pub delta_scatter_to: NativePort,
}

#[derive(Clone)]
pub(crate) enum BlockControlPorts {
    Attention {
        coordinates: NativePort,
        rotary: NativePort,
        visible: NativePort,
        fresh: NativePort,
        destinations: NativePort,
    },
    Recurrent {
        segments: NativePort,
    },
}

fn weight(
    graph: &mut NativeGraph,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightRole, NativePort)>,
) -> Result<seismic::WorkflowTensor, String> {
    let role = WeightRole { scope, kind };
    let plan = load
        .weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| format!("missing planned weight {role:?}"))?;
    let port = graph
        .port(plan.resident, &plan.shape)
        .map_err(|error| error.to_string())?;
    let tensor = port.tensor().clone();
    ports.push((role, port));
    Ok(tensor)
}

fn activation(geometry: &DecoderGeometry) -> Element {
    match geometry.activation_dtype {
        magnitude_model_contracts::ActivationDType::F16 => Element::f16(),
        magnitude_model_contracts::ActivationDType::BF16 => Element::bf16(),
    }
}

fn dense(
    graph: &mut NativeGraph,
    handle: &DenseKernels,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    residual: &WorkflowTensor,
    epsilon: f32,
) -> Result<WorkflowTensor, String> {
    let norm = weight(graph, load, scope, WeightKind::FeedForwardNorm, weights)?;
    let gate = weight(graph, load, scope, WeightKind::DenseGate, weights)?;
    let up = weight(graph, load, scope, WeightKind::DenseUp, weights)?;
    let down = weight(graph, load, scope, WeightKind::DenseDown, weights)?;
    let product = graph
        .enqueue(
            &handle.expand,
            qwen_dense_expand::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                gate_weight: (&gate).into(),
                up_weight: (&up).into(),
                eps: epsilon,
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    Ok(graph
        .enqueue(
            &handle.output,
            qwen_dense_output::WorkflowArgs {
                residual: residual.into(),
                product: (&product).into(),
                down_weight: (&down).into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value)
}

fn attention(
    graph: &mut NativeGraph,
    handle: &AttentionKernels,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    hidden: &WorkflowTensor,
    rows: u64,
    segments: u64,
    history_rows: u64,
    hidden_width: u64,
    heads: u64,
    kv_heads: u64,
    width: u64,
    rotary_width: u64,
    rotary_base: f32,
    epsilon: f32,
    activation: Element,
) -> Result<(WorkflowTensor, BlockStatePorts, BlockControlPorts), String> {
    let input_norm = weight(graph, load, scope, WeightKind::InputNorm, weights)?;
    let query_norm = weight(graph, load, scope, WeightKind::QueryNorm, weights)?;
    let key_norm = weight(graph, load, scope, WeightKind::KeyNorm, weights)?;
    let query_gate_weight = weight(graph, load, scope, WeightKind::QueryGate, weights)?;
    let key_weight = weight(graph, load, scope, WeightKind::Key, weights)?;
    let value_weight = weight(graph, load, scope, WeightKind::Value, weights)?;
    let output_weight = weight(graph, load, scope, WeightKind::AttentionOutput, weights)?;
    let group = heads / kv_heads;
    let normalize = graph
        .enqueue(
            &handle.normalize,
            qwen_attention_normalize::WorkflowArgs {
                hidden: hidden.into(),
                input_norm: (&input_norm).into(),
                epsilon,
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let projected = graph
        .enqueue(
            &handle.project,
            qwen_attention_project::WorkflowArgs {
                normalized: (&normalize).into(),
                query_norm: (&query_norm).into(),
                query_gate_weight: (&query_gate_weight).into(),
                key_weight: (&key_weight).into(),
                value_weight: (&value_weight).into(),
            },
        )
        .map_err(|error| error.to_string())?;
    let dimensions = [
        ("M", rows),
        ("KV", kv_heads),
        ("G", group),
        ("P", rotary_width / 2),
        ("S", width - rotary_width),
    ];
    let coordinates = graph
        .input_for(&handle.prepare, "coordinates", &dimensions)
        .map_err(|error| error.to_string())?;
    let rotary = graph
        .input_for(&handle.prepare, "rotary_components", &dimensions)
        .map_err(|error| error.to_string())?;
    let prepared = graph
        .enqueue(
            &handle.prepare,
            qwen_attention_prepare::WorkflowArgs {
                query_gate: (&projected.r0).into(),
                key: (&projected.r1).into(),
                query_norm: (&query_norm).into(),
                key_norm: (&key_norm).into(),
                coordinates: coordinates.tensor().into(),
                rotary_components: rotary.tensor().into(),
                base: rotary_base,
                epsilon,
            },
        )
        .map_err(|error| error.to_string())?;
    let attend_dimensions = [
        ("M", rows),
        ("T", history_rows),
        ("KV", kv_heads),
        ("G", group),
        ("W", width),
        ("R", segments),
    ];
    let visible = graph
        .input_for(&handle.attend, "visible", &attend_dimensions)
        .map_err(|error| error.to_string())?;
    let fresh = graph
        .input_for(&handle.attend, "fresh", &attend_dimensions)
        .map_err(|error| error.to_string())?;
    let mut accumulator = graph
        .local_for(&handle.attend, "accumulator", &attend_dimensions)
        .map_err(|error| error.to_string())?;
    let mut history_key = graph
        .port(activation, &[history_rows, kv_heads, width])
        .map_err(|error| error.to_string())?;
    let mut history_value = graph
        .port(activation, &[history_rows, kv_heads, width])
        .map_err(|error| error.to_string())?;
    let gated = graph
        .enqueue(
            &handle.attend,
            qwen_attention_attend::WorkflowArgs {
                query: (&prepared.r0).into(),
                prepared_key: (&prepared.r1).into(),
                value: (&projected.r2).into(),
                gate: (&prepared.r2).into(),
                visible: visible.tensor().into(),
                fresh: fresh.tensor().into(),
                history_key: history_key.tensor().into(),
                history_value: history_value.tensor().into(),
                accumulator: accumulator.tensor_mut().into(),
                scale: 1.0 / (width as f32).sqrt(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let destinations = graph
        .input_for(
            &handle.output,
            "destinations",
            &[
                ("M", rows),
                ("T", history_rows),
                ("KV", kv_heads),
                ("G", group),
                ("W", width),
                ("D", hidden_width),
            ],
        )
        .map_err(|error| error.to_string())?;
    let mixed = graph
        .enqueue(
            &handle.output,
            qwen_attention_output::WorkflowArgs {
                hidden: hidden.into(),
                gated: (&gated).into(),
                prepared_key: (&prepared.r1).into(),
                value: (&projected.r2).into(),
                output_weight: (&output_weight).into(),
                destinations: destinations.tensor().into(),
                history_key: history_key.tensor_mut().into(),
                history_value: history_value.tensor_mut().into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    Ok((
        mixed,
        BlockStatePorts::Attention {
            key: history_key,
            value: history_value,
        },
        BlockControlPorts::Attention {
            coordinates,
            rotary,
            visible,
            fresh,
            destinations,
        },
    ))
}

fn state_copy(
    graph: &mut NativeGraph,
    copy: &seismic::NativeKernel<copy_rows::Entry>,
    source: &WorkflowTensor,
    destination: &mut WorkflowTensor,
    source_rows: u64,
    destination_rows: u64,
    width: u64,
) -> Result<(NativePort, NativePort), String> {
    let dimensions = [
        ("N", 1),
        ("TS", source_rows),
        ("TD", destination_rows),
        ("KV", 1),
        ("W", width),
    ];
    let from = graph
        .input_for(copy, "from", &dimensions)
        .map_err(|e| e.to_string())?;
    let to = graph
        .input_for(copy, "to", &dimensions)
        .map_err(|e| e.to_string())?;
    let source_view = source.reshape(&[source_rows, 1, width]);
    let mut destination_view = destination.reshape(&[destination_rows, 1, width]);
    graph
        .enqueue(
            copy,
            copy_rows::WorkflowArgs {
                src: (&source_view).into(),
                dst: (&mut destination_view).into(),
                from: from.tensor().into(),
                to: to.tensor().into(),
            },
        )
        .map_err(|e| e.to_string())?;
    Ok((from, to))
}

fn recurrent(
    graph: &mut NativeGraph,
    handle: &RecurrentKernels,
    state_copies: &AttestedState,
    state: &StateResourcePlan,
    component_index: usize,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    hidden: &WorkflowTensor,
    rows: u64,
    slots: u64,
    hidden_width: u64,
    key_heads: u64,
    value_heads: u64,
    width: u64,
    convolution_width: u64,
    grouped: bool,
    epsilon: f32,
) -> Result<(WorkflowTensor, BlockStatePorts, BlockControlPorts), String> {
    let input_norm = weight(graph, load, scope, WeightKind::InputNorm, weights)?;
    let qkv_weight = weight(
        graph,
        load,
        scope,
        WeightKind::RecurrentQueryKeyValue,
        weights,
    )?;
    let gate_weight = weight(graph, load, scope, WeightKind::RecurrentGate, weights)?;
    let alpha_weight = weight(graph, load, scope, WeightKind::RecurrentAlpha, weights)?;
    let beta_weight = weight(graph, load, scope, WeightKind::RecurrentBeta, weights)?;
    let convolution = weight(
        graph,
        load,
        scope,
        WeightKind::RecurrentConvolution,
        weights,
    )?;
    let rate = weight(graph, load, scope, WeightKind::RecurrentDecay, weights)?;
    let time_bias = weight(graph, load, scope, WeightKind::RecurrentTimeBias, weights)?;
    let recurrent_norm = weight(graph, load, scope, WeightKind::RecurrentNorm, weights)?;
    let output_weight = weight(graph, load, scope, WeightKind::RecurrentOutput, weights)?;
    let channels = (2 * key_heads + value_heads) * width;
    let dimensions = [
        ("M", rows),
        ("B", slots),
        ("NK", key_heads),
        ("NV", value_heads),
        ("W", width),
        ("C", convolution_width),
    ];
    let mut window = graph
        .local_for(&handle.prepare, "window", &dimensions)
        .map_err(|error| error.to_string())?;
    let mut delta = graph
        .local_for(&handle.scan, "delta", &dimensions)
        .map_err(|error| error.to_string())?;
    let components = &state.target_state().recurrent_components;
    let window_spec = components
        .get(component_index)
        .ok_or("recurrent window state component is absent")?;
    let delta_spec = components
        .get(component_index + 1)
        .ok_or("recurrent delta state component is absent")?;
    let component_extents = |shape: &[usize]| -> Result<Vec<u64>, String> {
        shape
            .iter()
            .map(|extent| {
                u64::try_from(*extent).map_err(|_| "recurrent state extent exceeds u64".to_owned())
            })
            .collect()
    };
    let window_extents = component_extents(&window_spec.shape)?;
    let delta_extents = component_extents(&delta_spec.shape)?;
    let component_width = |extents: &[u64]| {
        extents.iter().try_fold(1u64, |value, extent| {
            value
                .checked_mul(*extent)
                .ok_or("recurrent state width overflows".to_owned())
        })
    };
    let window_width = component_width(&window_extents)?;
    let delta_width = component_width(&delta_extents)?;
    let window_element = Element::dense(window_spec.dtype);
    let delta_element = Element::dense(delta_spec.dtype);
    let copy_for = |element: Element| {
        state_copies
            .copies
            .iter()
            .find(|(candidate, _)| *candidate == element)
            .map(|(_, kernel)| kernel)
            .ok_or_else(|| format!("checked state copy kernel for {element:?} is absent"))
    };
    let window_copy = copy_for(window_element)?;
    let delta_copy = copy_for(delta_element)?;
    let mut state_slots = Vec::with_capacity(slots as usize);
    for _ in 0..slots {
        let previous_window = graph
            .port(window_element, &window_extents)
            .map_err(|e| e.to_string())?;
        let previous_delta = graph
            .port(delta_element, &delta_extents)
            .map_err(|e| e.to_string())?;
        let following_window = graph
            .port(window_element, &window_extents)
            .map_err(|e| e.to_string())?;
        let following_delta = graph
            .port(delta_element, &delta_extents)
            .map_err(|e| e.to_string())?;
        let (from, to) = state_copy(
            graph,
            window_copy,
            previous_window.tensor(),
            window.tensor_mut(),
            1,
            slots,
            window_width,
        )?;
        let (delta_from, delta_to) = state_copy(
            graph,
            delta_copy,
            previous_delta.tensor(),
            delta.tensor_mut(),
            1,
            slots,
            delta_width,
        )?;
        state_slots.push((
            previous_window,
            previous_delta,
            following_window,
            following_delta,
            from,
            to,
            delta_from,
            delta_to,
        ));
    }
    let normalized = graph
        .enqueue(
            &handle.normalize,
            qwen_recurrent_normalize::WorkflowArgs {
                hidden: hidden.into(),
                input_norm: (&input_norm).into(),
                epsilon,
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let projection = graph
        .enqueue(
            &handle.project,
            qwen_recurrent_project::WorkflowArgs {
                normalized: (&normalized).into(),
                qkv_weight: (&qkv_weight).into(),
                gate_weight: (&gate_weight).into(),
                alpha_weight: (&alpha_weight).into(),
                beta_weight: (&beta_weight).into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let segments = graph
        .input_for(&handle.prepare, "segments", &dimensions)
        .map_err(|error| error.to_string())?;
    let prepared = graph
        .enqueue(
            &handle.prepare,
            qwen_recurrent_prepare::WorkflowArgs {
                projection: (&projection).into(),
                convolution: (&convolution).into(),
                rate: (&rate).into(),
                time_bias: (&time_bias).into(),
                recurrent_norm: (&recurrent_norm).into(),
                segments: segments.tensor().into(),
                window: window.tensor().into(),
                preparation_epsilon: epsilon * width as f32,
            },
        )
        .map_err(|error| error.to_string())?;
    let scanned = graph
        .enqueue(
            &handle.scan,
            qwen_recurrent_scan::WorkflowArgs {
                prepared: (&prepared.r1).into(),
                decay: (&prepared.r2).into(),
                segments: segments.tensor().into(),
                delta: delta.tensor().into(),
                grouped,
            },
        )
        .map_err(|error| error.to_string())?;
    let gated = graph
        .enqueue(
            &handle.mix,
            qwen_recurrent_mix::WorkflowArgs {
                projection: (&projection).into(),
                mixed: (&scanned.r1).into(),
                recurrent_norm: (&recurrent_norm).into(),
                epsilon,
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let mixed = graph
        .enqueue(
            &handle.output,
            qwen_recurrent_output::WorkflowArgs {
                hidden: hidden.into(),
                gated: (&gated).into(),
                output_weight: (&output_weight).into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let mut slot_ports = Vec::with_capacity(state_slots.len());
    for (
        previous_window,
        previous_delta,
        mut following_window,
        mut following_delta,
        window_from,
        window_to,
        delta_from,
        delta_to,
    ) in state_slots
    {
        let (window_scatter_from, window_scatter_to) = state_copy(
            graph,
            window_copy,
            &prepared.r0,
            following_window.tensor_mut(),
            slots,
            1,
            window_width,
        )?;
        let (delta_scatter_from, delta_scatter_to) = state_copy(
            graph,
            delta_copy,
            &scanned.r0,
            following_delta.tensor_mut(),
            slots,
            1,
            delta_width,
        )?;
        slot_ports.push(RecurrentSlotPorts {
            previous_window,
            previous_delta,
            following_window,
            following_delta,
            window_from,
            window_to,
            delta_from,
            delta_to,
            window_scatter_from,
            window_scatter_to,
            delta_scatter_from,
            delta_scatter_to,
        });
    }
    let _ = hidden_width;
    Ok((
        mixed,
        BlockStatePorts::Recurrent { slots: slot_ports },
        BlockControlPorts::Recurrent { segments },
    ))
}

fn routed(
    graph: &mut NativeGraph,
    handle: &RoutedKernels,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    residual: &WorkflowTensor,
    rows: u64,
    hidden: u64,
    experts: u64,
    selected: u64,
    normalize_selected: bool,
    epsilon: f32,
) -> Result<(WorkflowTensor, NativePort), String> {
    let norm = weight(graph, load, scope, WeightKind::FeedForwardNorm, weights)?;
    let router_weight = weight(graph, load, scope, WeightKind::Router, weights)?;
    let shared_router = weight(graph, load, scope, WeightKind::SharedRouter, weights)?;
    let expert_gate = weight(graph, load, scope, WeightKind::ExpertGate, weights)?;
    let expert_up = weight(graph, load, scope, WeightKind::ExpertUp, weights)?;
    let expert_down = weight(graph, load, scope, WeightKind::ExpertDown, weights)?;
    let shared_gate = weight(graph, load, scope, WeightKind::SharedGate, weights)?;
    let shared_up = weight(graph, load, scope, WeightKind::SharedUp, weights)?;
    let shared_down = weight(graph, load, scope, WeightKind::SharedDown, weights)?;
    let source_rows = graph
        .input_for(
            &handle.normalize,
            "source_rows",
            &[("M", rows), ("O", rows), ("H", hidden)],
        )
        .map_err(|error| error.to_string())?;
    let normalized = graph
        .enqueue(
            &handle.normalize,
            qwen_routed_normalize::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                source_rows: source_rows.tensor().into(),
                eps: epsilon,
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let logits = graph
        .enqueue(
            &handle.logits,
            qwen_routed_logits::WorkflowArgs {
                normalized: (&normalized).into(),
                router_weight: (&router_weight).into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let dimensions = [("O", rows), ("E", experts), ("K", selected)];
    let mut routes = graph
        .local_for(&handle.select, "routes", &dimensions)
        .map_err(|error| error.to_string())?;
    let mut scores = graph
        .local_for(&handle.select, "scores", &dimensions)
        .map_err(|error| error.to_string())?;
    graph
        .enqueue(
            &handle.select,
            qwen_routed_select::WorkflowArgs {
                logits: (&logits).into(),
                selected: i32::from(normalize_selected),
                routes: routes.tensor_mut().into(),
                scores: scores.tensor_mut().into(),
            },
        )
        .map_err(|error| error.to_string())?;
    let expanded = graph
        .enqueue(
            &handle.expand,
            qwen_routed_expand::WorkflowArgs {
                normalized: (&normalized).into(),
                expert_gate: (&expert_gate).into(),
                expert_up: (&expert_up).into(),
                shared_gate: (&shared_gate).into(),
                shared_up: (&shared_up).into(),
                shared_control: (&shared_router).into(),
                routes: routes.tensor().into(),
            },
        )
        .map_err(|error| error.to_string())?;
    let output = graph
        .enqueue(
            &handle.output,
            qwen_routed_output::WorkflowArgs {
                residual: residual.into(),
                source_rows: source_rows.tensor().into(),
                expert_product: (&expanded.r0).into(),
                shared_product: (&expanded.r1).into(),
                shared_coefficient: (&expanded.r2).into(),
                expert_down: (&expert_down).into(),
                shared_down: (&shared_down).into(),
                routes: routes.tensor().into(),
                scores: scores.tensor().into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    Ok((output, source_rows))
}

impl PreparedTargetBlockGraph {
    pub fn prepare(
        device: &Device,
        handle: &AttestedTargetBlock,
        state_copies: &AttestedState,
        load: &ModelLoadPlan,
        geometry: &DecoderGeometry,
        state: &StateResourcePlan,
        block_index: usize,
        rows: u64,
        segments: u64,
        slots: u64,
        history_rows: u64,
    ) -> Result<Self, String> {
        let block = geometry
            .blocks
            .get(block_index)
            .ok_or_else(|| "target block geometry is absent".to_owned())?;
        let scope = WeightScope::TargetBlock(
            u32::try_from(block_index).map_err(|_| "target block index exceeds u32")?,
        );
        let mut graph = device.native_graph();
        let mut weights = Vec::new();
        let hidden = graph
            .port(Element::f32(), &[rows, geometry.hidden])
            .map_err(|error| error.to_string())?;
        let activation = activation(geometry);
        let component_index = geometry.blocks[..block_index]
            .iter()
            .filter(|block| matches!(block.mixer, MixerGeometry::Recurrent(_)))
            .count()
            * 2;
        let (mixed, state, controls) = match (&block.mixer, &handle.mixer) {
            (MixerGeometry::Attention(shape), AttestedMixer::Attention(kernels)) => {
                let RotarySemantics::Interleaved {
                    width: rotary_width,
                    base,
                    ..
                } = &shape.rotary;
                attention(
                    &mut graph,
                    kernels,
                    load,
                    scope,
                    &mut weights,
                    hidden.tensor(),
                    rows,
                    segments,
                    history_rows,
                    geometry.hidden,
                    shape.heads,
                    shape.kv_heads,
                    shape.width,
                    *rotary_width,
                    *base as f32,
                    geometry.epsilon as f32,
                    activation,
                )?
            }
            (MixerGeometry::Recurrent(shape), AttestedMixer::Recurrent(kernels)) => recurrent(
                &mut graph,
                kernels,
                state_copies,
                state,
                component_index,
                load,
                scope,
                &mut weights,
                hidden.tensor(),
                rows,
                slots,
                geometry.hidden,
                shape.key_heads,
                shape.value_heads,
                shape.width,
                shape.convolution_width,
                matches!(shape.head_mapping, RecurrentHeadMapping::Grouped),
                geometry.epsilon as f32,
            )?,
            _ => return Err("target block mixer and prepared kernel disagree".into()),
        };
        let (output, routed_rows) = match (&block.feedforward, &handle.feed_forward) {
            (FeedForwardGeometry::Dense { .. }, AttestedFeedForward::Dense(kernels)) => (
                dense(
                    &mut graph,
                    kernels,
                    load,
                    scope,
                    &mut weights,
                    &mixed,
                    geometry.epsilon as f32,
                )?,
                None,
            ),
            (FeedForwardGeometry::Routed(shape), AttestedFeedForward::Routed(kernels)) => {
                let (output, source_rows) = routed(
                    &mut graph,
                    kernels,
                    load,
                    scope,
                    &mut weights,
                    &mixed,
                    rows,
                    geometry.hidden,
                    shape.count,
                    shape.selected,
                    shape.normalize_selected,
                    geometry.epsilon as f32,
                )?;
                (output, Some(source_rows))
            }
            _ => return Err("target block feed-forward and prepared kernel disagree".into()),
        };
        graph.export(&output).map_err(|error| error.to_string())?;
        let plan = graph.seal().map_err(|error| error.to_string())?;
        Ok(Self {
            plan,
            hidden,
            weights,
            state,
            controls,
            routed_rows,
            output,
        })
    }
}
