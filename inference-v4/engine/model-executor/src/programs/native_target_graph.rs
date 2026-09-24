//! Seismic-owned decoder block graphs. The engine supplies semantic model
//! topology and external state; checked entry contracts derive every stage
//! edge and intermediate allocation.

use crate::{
    ModelLoadPlan, ResidentBlockWeights, ResidentFeedForwardWeights, ResidentMixerWeights,
    ResidentTarget, ResidentWeight, ResourceLimits, StateResourcePlan,
    native::{
        AttestedFeedForward, AttestedMixer, AttestedTarget, AttestedTargetBlock,
    },
    programs::graph::attention::{AttentionBlock, AttentionWeights, attention},
    programs::graph::dense::dense,
    programs::graph::recurrent::{
        RecurrentBlock, RecurrentControlPorts, RecurrentStatePorts, recurrent,
    },
    programs::native_constants::{ConstantTensors, GraphConstant},
};
use magnitude_model_batching::MAX_CLASS_SEGMENTS;
use magnitude_model_contracts::{
    DecoderGeometry, FeedForwardGeometry, MixerGeometry, RecurrentHeadMapping, WeightKind,
    WeightRole, WeightScope,
};
use magnitude_model_kernels::{
    qwen_embedding_rows, };
use seismic::{
    BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphFamily, NativeGraphPlan,
    NativePort, WorkflowTensor,
};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone)]
pub struct PreparedTargetGraphs {
    entries: BTreeMap<u64, PreparedTargetEntryGraph>,
    classes: BTreeMap<(u64, u64, u64), Vec<PreparedTargetBlockGraph>>,
    family: NativeGraphFamily,
    max_output_bytes: u64,
    /// Decoder blocks, each one graph run per step.
    blocks: usize,
    seal: SealReport,
}

/// Sealing cost of a graph set: shape classes, graphs actually sealed after
/// sharing identical block plans, and wall-clock seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SealReport {
    pub classes: usize,
    pub sealed_graphs: usize,
    pub seconds: f64,
}

/// The inputs that determine a block's sealed graph apart from its layer
/// index: its geometry, the resident element and shape of each weight kind,
/// and its recurrent state components. Blocks with equal shapes share plans.
#[derive(Debug, PartialEq)]
struct BlockGraphShape {
    geometry: String,
    weights: Vec<(WeightKind, Element, Vec<u64>)>,
    state: String,
}

fn block_graph_shapes(
    load: &ModelLoadPlan,
    geometry: &DecoderGeometry,
    state: &StateResourcePlan,
) -> Result<Vec<BlockGraphShape>, String> {
    let components = &state.target_state().recurrent_components;
    let mut recurrent_index = 0usize;
    geometry
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            let scope = WeightScope::TargetBlock(
                u32::try_from(index).map_err(|_| "target block index exceeds u32")?,
            );
            let mut weights = load
                .weights()
                .filter(|weight| weight.role.scope == scope)
                .map(|weight| (weight.role.kind, weight.resident, weight.shape.clone()))
                .collect::<Vec<_>>();
            weights.sort_by_key(|(kind, _, _)| format!("{kind:?}"));
            let state = match block.mixer {
                MixerGeometry::Recurrent(_) => {
                    let first = recurrent_index * 2;
                    recurrent_index += 1;
                    format!("{:?}", components.get(first..first + 2))
                }
                MixerGeometry::Attention(_) => String::new(),
            };
            Ok(BlockGraphShape {
                geometry: format!("{block:?}"),
                weights,
                state,
            })
        })
        .collect()
}

impl PreparedTargetGraphs {
    pub(crate) fn prepare(
        device: &Device,
        handles: &AttestedTarget,
        load: &ModelLoadPlan,
        geometry: &DecoderGeometry,
        state: &StateResourcePlan,
        limits: ResourceLimits,
    ) -> Result<Self, String> {
        let row_classes = magnitude_model_batching::row_classes(limits.max_batch_rows);
        if row_classes.is_empty() {
            return Err(format!(
                "target batch row bound {} has no row class",
                limits.max_batch_rows
            ));
        }
        let max_slots = u64::try_from(limits.in_flight_requests)
            .map_err(|_| "target request slot bound exceeds u64")?;
        let history_rows = u64::try_from(state.target_state().history_rows)
            .map_err(|_| "target history row bound exceeds u64")?;
        // Blocks whose sealed graph would be identical share one plan per
        // class: the plan names weights by kind only, and each block binds
        // its own resident weights to it.
        let shapes = block_graph_shapes(load, geometry, state)?;
        let mut classes = BTreeMap::new();
        let mut entries = BTreeMap::new();
        let mut plans = Vec::new();
        let mut max_output_bytes = 0u64;
        let trace = std::env::var_os("MAGNITUDE_V4_TRACE_GRAPH_SEAL").is_some();
        let began_all = Instant::now();
        let mut sealed_graphs = 0usize;
        for rows in row_classes.into_iter().map(|rows| rows as u64) {
            let entry = PreparedTargetEntryGraph::prepare(
                device,
                &handles.embedding,
                load,
                geometry,
                rows,
            )?;
            sealed_graphs += 1;
            max_output_bytes = max_output_bytes.max(entry.plan.output_bytes());
            plans.push(entry.plan.clone());
            entries.insert(rows, entry);
            let mut segments = 1u64;
            while segments <= MAX_CLASS_SEGMENTS as u64 {
                for slots in 1..=max_slots {
                    let began = Instant::now();
                    let mut sealed: Vec<(usize, PreparedTargetBlockGraph)> = Vec::new();
                    let mut blocks = Vec::with_capacity(handles.blocks.len());
                    for (index, handle) in handles.blocks.iter().enumerate() {
                        if let Some((_, shared)) = sealed
                            .iter()
                            .find(|(first, _)| shapes[*first] == shapes[index])
                        {
                            blocks.push(shared.clone());
                            continue;
                        }
                        let block = PreparedTargetBlockGraph::prepare(device, handle,
                            load, geometry, state, index, rows, segments, slots, history_rows)
                            .map_err(|error| format!(
                                "target graph row class {rows}, history segments {segments}, request slots {slots}, block {index}: {error}"
                            ))?;
                        sealed_graphs += 1;
                        max_output_bytes = max_output_bytes.max(block.plan.output_bytes());
                        plans.push(block.plan.clone());
                        sealed.push((index, block.clone()));
                        blocks.push(block);
                    }
                    classes.insert((rows, segments, slots), blocks);
                    if trace {
                        eprintln!(
                            "target graph sealed rows={rows} segments={segments} slots={slots} blocks={} distinct={} elapsed_ms={:.3}",
                            handles.blocks.len(),
                            sealed.len(),
                            began.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
                segments *= 2;
            }
        }
        let seal_seconds = began_all.elapsed().as_secs_f64();
        let family = NativeGraphFamily::new(&plans).map_err(|error| error.to_string())?;
        let seal = SealReport {
            classes: classes.len(),
            sealed_graphs,
            seconds: seal_seconds,
        };
        Ok(Self {
            entries,
            classes,
            family,
            max_output_bytes,
            blocks: handles.blocks.len(),
            seal,
        })
    }

    /// How many class graphs sealing formed and how long it took.
    pub fn seal_report(&self) -> SealReport {
        self.seal
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub fn output_bytes(&self) -> u64 {
        self.max_output_bytes
    }

    /// Graph runs one step queues on a workspace slot before any completes:
    /// the embedding entry and every block.
    pub fn runs_per_step(&self) -> usize {
        1 + self.blocks
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
        let mut constants = ConstantTensors::new(resident.embedding.tensor().device());
        let mut bound = BTreeMap::new();
        for (class, blocks) in &self.classes {
            let mut class_bound = Vec::with_capacity(blocks.len());
            for (index, graph) in blocks.iter().enumerate() {
                let block = resident
                    .blocks
                    .get(index)
                    .ok_or("resident target block missing")?;
                let constant_tensors = graph
                    .constants
                    .iter()
                    .map(|constant| Ok((constant.port(), constants.tensor(constant)?)))
                    .collect::<Result<Vec<_>, String>>()?;
                let fixed = graph
                    .weights
                    .iter()
                    .map(|(role, port)| {
                        let weight = block_weight(block, role.kind)?;
                        Ok((port, weight.tensor()))
                    })
                    .chain(constant_tensors.iter().map(|(port, tensor)| Ok((*port, tensor))))
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
    /// Host constants bound statically with the weights.
    pub constants: Vec<GraphConstant>,
    pub state: BlockStatePorts,
    pub controls: BlockControlPorts,
    pub output: WorkflowTensor,
}

#[derive(Clone)]
pub(crate) enum BlockStatePorts {
    Attention { key: NativePort, value: NativePort },
    Recurrent(RecurrentStatePorts),
}

#[derive(Clone)]
pub(crate) enum BlockControlPorts {
    Attention {
        coordinates: NativePort,
        visible: NativePort,
        fresh: NativePort,
        destinations: NativePort,
    },
    Recurrent(RecurrentControlPorts),
}

pub(crate) fn weight(
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

impl PreparedTargetBlockGraph {
    pub fn prepare(
        device: &Device,
        handle: &AttestedTargetBlock,
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
        let mut constants = Vec::new();
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
                let mut weight = |kind| weight(&mut graph, load, scope, kind, &mut weights);
                let attention_weights = AttentionWeights {
                    input_norm: weight(WeightKind::InputNorm)?,
                    query_norm: weight(WeightKind::QueryNorm)?,
                    key_norm: weight(WeightKind::KeyNorm)?,
                    query_gate: weight(WeightKind::QueryGate)?,
                    key: weight(WeightKind::Key)?,
                    value: weight(WeightKind::Value)?,
                    output: weight(WeightKind::AttentionOutput)?,
                };
                let (mixed, state, controls) = attention(
                    &mut graph,
                    kernels,
                    &attention_weights,
                    &mut constants,
                    hidden.tensor(),
                    AttentionBlock {
                        rows,
                        segments,
                        history_rows,
                        heads: shape.heads,
                        kv_heads: shape.kv_heads,
                        width: shape.width,
                        rotary: &shape.rotary,
                        epsilon: geometry.epsilon as f32,
                        activation,
                    },
                )?;
                (
                    mixed,
                    BlockStatePorts::Attention {
                        key: state.key,
                        value: state.value,
                    },
                    BlockControlPorts::Attention {
                        coordinates: controls.coordinates,
                        visible: controls.visible,
                        fresh: controls.fresh,
                        destinations: controls.destinations,
                    },
                )
            }
            (MixerGeometry::Recurrent(shape), AttestedMixer::Recurrent(kernels)) => {
                let (mixed, state, controls) = recurrent(
                    &mut graph,
                    kernels,
                    state,
                    load,
                    scope,
                    &mut weights,
                    hidden.tensor(),
                    RecurrentBlock {
                        rows,
                        slots,
                        key_heads: shape.key_heads,
                        value_heads: shape.value_heads,
                        width: shape.width,
                        convolution_width: shape.convolution_width,
                        grouped: matches!(shape.head_mapping, RecurrentHeadMapping::Grouped),
                        epsilon: geometry.epsilon as f32,
                        component_index,
                    },
                )?;
                (
                    mixed,
                    BlockStatePorts::Recurrent(state),
                    BlockControlPorts::Recurrent(controls),
                )
            }
            _ => return Err("target block mixer and prepared kernel disagree".into()),
        };
        let output = match (&block.feedforward, &handle.feed_forward) {
            (FeedForwardGeometry::Dense { .. }, AttestedFeedForward::Dense(kernels)) => dense(
                &mut graph,
                kernels,
                load,
                scope,
                &mut weights,
                &mut constants,
                &mixed,
                rows,
                geometry.epsilon as f32,
            )?,
            (FeedForwardGeometry::Routed(shape), AttestedFeedForward::Routed(kernels)) => {
                super::graph::routed::routed(
                    &mut graph,
                    kernels,
                    load,
                    scope,
                    &mut weights,
                    &mixed,
                    rows,
                    geometry.hidden,
                    shape,
                    geometry.epsilon as f32,
                )?
            }
            _ => return Err("target block feed-forward and prepared kernel disagree".into()),
        };
        graph.export(&output).map_err(|error| error.to_string())?;
        let plan = graph.seal().map_err(|error| error.to_string())?;
        Ok(Self {
            plan,
            hidden,
            weights,
            constants,
            state,
            controls,
            output,
        })
    }
}

