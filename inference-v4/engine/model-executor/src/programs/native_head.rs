//! Single-block native draft-head program over validated row and feature
//! launches. Checked results are written into the planned head pools.

use super::{HeadProgram, ProjectGraphOutput, ReadySubmission};
use crate::{
    DeviceError, GraphOutputTensor, HeadLaunchCore, InvariantError, ModelLoadPlan,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, ProjectLaunchCore, ResidentHead,
    ResidentWeight, SubmitError, ValidatedHeadLaunch, ValidatedProjectionLaunch,
    native::AttestedHead,
};
use magnitude_model_contracts::{
    ActivationDType, AttentionGeometry, DecoderGeometry, RotarySemantics, WeightKind, WeightRole,
    WeightScope,
};
use magnitude_model_kernels::{
    copy_rows, head_logits_rows, qwen_attention_attend, qwen_attention_normalize,
    qwen_attention_output, qwen_attention_prepare, qwen_attention_project, qwen_dense_expand,
    qwen_dense_output, qwen_features_rows, qwen_head_rows, sample_rows, shape_rows,
};
use magnitude_model_state::{LayerRef, PlaneName, VectorKind};
use seismic::{
    BoundNativeGraphPlan, Device, Element, NativeGraphBindings, NativeGraphFamily,
    NativeGraphFamilySlot, NativeGraphPlan, NativeKernel, NativePort, Tensor, WorkflowTensor,
};
use std::rc::Rc;

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native head program",
        detail: detail.into(),
    })
}
fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn activation(dtype: ActivationDType) -> Element {
    match dtype {
        ActivationDType::F16 => Element::f16(),
        ActivationDType::BF16 => Element::bf16(),
    }
}

fn planned_head_weight(
    graph: &mut seismic::NativeGraph,
    load: &ModelLoadPlan,
    role: WeightRole,
    weights: &mut Vec<(WeightRole, NativePort)>,
) -> Result<WorkflowTensor, SubmitError> {
    let plan = load
        .weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| invalid(format!("planned head weight {role:?} is absent")))?;
    let port = graph.port(plan.resident, &plan.shape).map_err(device)?;
    let tensor = port.tensor().clone();
    weights.push((role, port));
    Ok(tensor)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadProjectGraphClass {
    pub rows: u64,
}

pub(crate) struct PreparedHeadProjectGraphs {
    variants: Vec<PreparedHeadProjectGraph>,
    family: NativeGraphFamily,
}

struct PreparedHeadProjectGraph {
    class: HeadProjectGraphClass,
    plan: NativeGraphPlan,
    features: Vec<NativePort>,
    features_from: Vec<NativePort>,
    features_to: Vec<NativePort>,
    shaping: NativePort,
    history: NativePort,
    mask: NativePort,
    draws: NativePort,
    weights: Vec<(WeightRole, NativePort)>,
    logits: WorkflowTensor,
    selected: NativePort,
}

pub(crate) struct HeadProjectGraphInputs<'a> {
    pub features: &'a [Tensor],
    pub shaping: &'a [u8],
    pub history: &'a [u8],
    pub mask: &'a [u8],
    pub draws: &'a [u8],
}

pub(crate) struct HeadProjectGraphResult {
    pub logits: GraphOutputTensor,
    pub selected: GraphOutputTensor,
}

impl PreparedHeadProjectGraphs {
    pub(crate) fn prepare(
        target_device: &Device,
        handles: &AttestedHead,
        shape: &NativeKernel<shape_rows::Entry>,
        sample: &NativeKernel<sample_rows::Entry>,
        load: &ModelLoadPlan,
        geometry: &DecoderGeometry,
        copy: &NativeKernel<copy_rows::Entry>,
        classes: impl IntoIterator<Item = HeadProjectGraphClass>,
    ) -> Result<Self, SubmitError> {
        let handle = handles
            .blocks
            .first()
            .ok_or_else(|| invalid("attested projection slot is absent"))?;
        let mut variants = Vec::new();
        for class in classes {
            if class.rows == 0 {
                return Err(invalid("head project graph has no rows"));
            }
            if variants
                .iter()
                .any(|variant: &PreparedHeadProjectGraph| variant.class == class)
            {
                return Err(invalid("head project graph class is duplicated"));
            }
            let vocabulary = geometry.vocabulary;
            let mut graph = target_device.native_graph();
            let copy_dims = [
                ("N", 1),
                ("TS", 1),
                ("TD", class.rows),
                ("KV", 1),
                ("W", geometry.hidden),
            ];
            let mut features_local = graph.local_for(copy, "dst", &copy_dims).map_err(device)?;
            let mut features = Vec::with_capacity(class.rows as usize);
            let mut features_from = Vec::with_capacity(class.rows as usize);
            let mut features_to = Vec::with_capacity(class.rows as usize);
            for _ in 0..class.rows {
                let source = graph
                    .port(
                        activation(geometry.activation_dtype),
                        &[1, 1, geometry.hidden],
                    )
                    .map_err(device)?;
                let from = graph.input_for(copy, "from", &copy_dims).map_err(device)?;
                let to = graph.input_for(copy, "to", &copy_dims).map_err(device)?;
                graph
                    .enqueue(
                        copy,
                        copy_rows::WorkflowArgs {
                            src: source.tensor().into(),
                            dst: features_local.tensor_mut().into(),
                            from: from.tensor().into(),
                            to: to.tensor().into(),
                        },
                    )
                    .map_err(device)?;
                features.push(source);
                features_from.push(from);
                features_to.push(to);
            }
            let shaping = graph
                .input_for(
                    shape,
                    "params",
                    &[("Sx", class.rows), ("V", vocabulary), ("Hn", 64)],
                )
                .map_err(device)?;
            let history = graph
                .input_for(
                    shape,
                    "history",
                    &[("Sx", class.rows), ("V", vocabulary), ("Hn", 64)],
                )
                .map_err(device)?;
            let mask = graph
                .input_for(sample, "mask", &[("M", class.rows), ("V", vocabulary)])
                .map_err(device)?;
            let draws = graph
                .input_for(sample, "draws", &[("M", class.rows), ("V", vocabulary)])
                .map_err(device)?;
            let mut weights = Vec::new();
            let output_weight = planned_head_weight(
                &mut graph,
                load,
                WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::Output,
                },
                &mut weights,
            )?;
            let logits = graph
                .enqueue(
                    &handle.logits,
                    head_logits_rows::WorkflowArgs {
                        features: features_local.tensor().into(),
                        weight: (&output_weight).into(),
                    },
                )
                .map_err(device)?
                .value;
            let mut shaped = graph
                .local_for(
                    shape,
                    "out",
                    &[("Sx", class.rows), ("V", vocabulary), ("Hn", 64)],
                )
                .map_err(device)?;
            graph
                .enqueue(
                    shape,
                    shape_rows::WorkflowArgs {
                        logits: (&logits).into(),
                        params: shaping.tensor().into(),
                        history: history.tensor().into(),
                        out: shaped.tensor_mut().into(),
                    },
                )
                .map_err(device)?;
            let mut selected = graph
                .local_for(sample, "result", &[("M", class.rows), ("V", vocabulary)])
                .map_err(device)?;
            graph
                .enqueue(
                    sample,
                    sample_rows::WorkflowArgs {
                        logits: shaped.tensor().into(),
                        mask: mask.tensor().into(),
                        draws: draws.tensor().into(),
                        result: selected.tensor_mut().into(),
                    },
                )
                .map_err(device)?;
            graph.export(&logits).map_err(device)?;
            graph.export(selected.tensor()).map_err(device)?;
            let plan = graph.seal().map_err(device)?;
            variants.push(PreparedHeadProjectGraph {
                class,
                plan,
                features,
                features_from,
                features_to,
                shaping,
                history,
                mask,
                draws,
                weights,
                logits,
                selected,
            });
        }
        if variants.is_empty() {
            return Err(invalid("head project graph family has no classes"));
        }
        let plans = variants
            .iter()
            .map(|variant| variant.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self { variants, family })
    }

    pub(crate) fn workspace_bytes_max(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub(crate) fn output_bytes_max(&self) -> u64 {
        self.family.output_bytes()
    }

    pub(crate) fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub(crate) fn plans(&self) -> impl Iterator<Item = (HeadProjectGraphClass, &NativeGraphPlan)> {
        self.variants
            .iter()
            .map(|variant| (variant.class, &variant.plan))
    }

    pub(crate) fn plan(
        &self,
        class: HeadProjectGraphClass,
    ) -> Result<&NativeGraphPlan, SubmitError> {
        Ok(&self.variant(class)?.plan)
    }

    pub(crate) fn bindings(
        &self,
        class: HeadProjectGraphClass,
        inputs: HeadProjectGraphInputs<'_>,
        bound: &BoundNativeGraphPlan,
    ) -> Result<NativeGraphBindings, SubmitError> {
        let variant = self.variant(class)?;
        let mut bindings = bound.bindings();
        if inputs.features.len() != variant.features.len() {
            return Err(invalid(
                "head project feature bindings differ from graph class",
            ));
        }
        for (port, tensor) in variant.features.iter().zip(inputs.features) {
            bindings.set(port, tensor).map_err(device)?;
        }
        Ok(bindings)
    }

    pub(crate) fn run(
        &self,
        class: HeadProjectGraphClass,
        slot: &mut NativeGraphFamilySlot,
        bindings: NativeGraphBindings,
        mut output: NativeGraphOutputLease,
        inputs: HeadProjectGraphInputs<'_>,
    ) -> Result<HeadProjectGraphResult, SubmitError> {
        let variant = self.variant(class)?;
        let mut active = slot.activate(&variant.plan).map_err(device)?;
        for (row, (from, to)) in variant
            .features_from
            .iter()
            .zip(&variant.features_to)
            .enumerate()
        {
            active
                .write_input(from, &0_i32.to_le_bytes())
                .map_err(device)?;
            active
                .write_input(to, &(row as i32).to_le_bytes())
                .map_err(device)?;
        }
        active
            .write_input(&variant.shaping, inputs.shaping)
            .map_err(device)?;
        active
            .write_input(&variant.history, inputs.history)
            .map_err(device)?;
        active
            .write_input(&variant.mask, inputs.mask)
            .map_err(device)?;
        active
            .write_input(&variant.draws, inputs.draws)
            .map_err(device)?;
        let outputs = output
            .activate(&variant.plan)
            .map_err(SubmitError::Invariant)?;
        let outputs = active
            .attach(bindings, outputs)
            .and_then(super::run_graph)
            .map_err(device)?;
        let owner = output.publish(outputs);
        let logits = owner
            .tensor(&variant.logits)
            .ok_or_else(|| invalid("head project graph omitted retained logits"))?;
        let selected = owner
            .tensor(variant.selected.tensor())
            .ok_or_else(|| invalid("head project graph omitted retained selection"))?;
        Ok(HeadProjectGraphResult { logits, selected })
    }

    fn variant(
        &self,
        class: HeadProjectGraphClass,
    ) -> Result<&PreparedHeadProjectGraph, SubmitError> {
        self.variants
            .iter()
            .find(|variant| variant.class == class)
            .ok_or_else(|| invalid("head project graph class was not prepared"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadForwardGraphClass {
    pub rows: u64,
    pub segments: u64,
    pub history_rows: u64,
}

struct PreparedHeadForwardGraph {
    class: HeadForwardGraphClass,
    plan: NativeGraphPlan,
    tokens: NativePort,
    conditioning: Vec<NativePort>,
    conditioning_from: Vec<NativePort>,
    conditioning_to: Vec<NativePort>,
    coordinates: NativePort,
    rotary: NativePort,
    visible: NativePort,
    fresh: NativePort,
    destinations: NativePort,
    history_key: NativePort,
    history_value: NativePort,
    out_rows: NativePort,
    weights: Vec<(WeightRole, NativePort)>,
    features: WorkflowTensor,
}

pub(crate) struct PreparedHeadForwardGraphs {
    variants: Vec<PreparedHeadForwardGraph>,
    family: NativeGraphFamily,
}

pub(crate) struct HeadForwardGraphInputs<'a> {
    pub tokens: &'a [u8],
    pub conditioning: &'a [Tensor],
    pub coordinates: &'a [u8],
    pub rotary: &'a [u8],
    pub visible: &'a [u8],
    pub fresh: &'a [u8],
    pub destinations: &'a [u8],
    pub history_key: &'a Tensor,
    pub history_value: &'a Tensor,
    pub out_rows: &'a [u8],
}

pub(crate) struct HeadForwardGraphResult {
    pub features: GraphOutputTensor,
}

/// One admitted head lane. Forward and projection are mutually exclusive
/// submissions, so all exact variants share one maximal Seismic arena.
pub struct PreparedHeadGraphs {
    forward: PreparedHeadForwardGraphs,
    project: PreparedHeadProjectGraphs,
    family: NativeGraphFamily,
}

pub(crate) struct BoundHeadGraphs {
    prepared: Rc<PreparedHeadGraphs>,
    forward: Vec<BoundNativeGraphPlan>,
    project: Vec<BoundNativeGraphPlan>,
}

impl PreparedHeadGraphs {
    pub(crate) fn from_parts(
        forward: PreparedHeadForwardGraphs,
        project: PreparedHeadProjectGraphs,
    ) -> Result<Self, SubmitError> {
        let plans = forward
            .plans()
            .map(|(_, plan)| plan.clone())
            .chain(project.plans().map(|(_, plan)| plan.clone()))
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self {
            forward,
            project,
            family,
        })
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

    pub(crate) fn forward(&self) -> &PreparedHeadForwardGraphs {
        &self.forward
    }

    pub(crate) fn project(&self) -> &PreparedHeadProjectGraphs {
        &self.project
    }

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        resident: &ResidentHead,
    ) -> Result<BoundHeadGraphs, SubmitError> {
        let bind = |plan: &NativeGraphPlan,
                    weights: &[(WeightRole, NativePort)]|
         -> Result<BoundNativeGraphPlan, SubmitError> {
            let fixed = weights
                .iter()
                .map(|(role, port)| Ok((port, resident_head_weight(resident, *role)?.tensor())))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            plan.bind_static(&fixed).map_err(device)
        };
        let forward = self
            .forward
            .variants
            .iter()
            .map(|variant| bind(&variant.plan, &variant.weights))
            .collect::<Result<Vec<_>, _>>()?;
        let project = self
            .project
            .variants
            .iter()
            .map(|variant| bind(&variant.plan, &variant.weights))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BoundHeadGraphs {
            prepared: self.clone(),
            forward,
            project,
        })
    }
}

impl BoundHeadGraphs {
    pub(crate) fn forward(
        &self,
        class: HeadForwardGraphClass,
    ) -> Result<(&PreparedHeadForwardGraphs, &BoundNativeGraphPlan), SubmitError> {
        let index = self
            .prepared
            .forward
            .variants
            .iter()
            .position(|variant| variant.class == class)
            .ok_or_else(|| invalid("head forward graph class was not prepared"))?;
        Ok((&self.prepared.forward, &self.forward[index]))
    }

    pub(crate) fn project(
        &self,
        class: HeadProjectGraphClass,
    ) -> Result<(&PreparedHeadProjectGraphs, &BoundNativeGraphPlan), SubmitError> {
        let index = self
            .prepared
            .project
            .variants
            .iter()
            .position(|variant| variant.class == class)
            .ok_or_else(|| invalid("head project graph class was not prepared"))?;
        Ok((&self.prepared.project, &self.project[index]))
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
                WeightKind::DenseGate => &block.feedforward.gate,
                WeightKind::DenseUp => &block.feedforward.up,
                WeightKind::DenseDown => &block.feedforward.down,
                WeightKind::OutputNorm => &block.output_norm,
                _ => return Err(invalid(format!("head weight role {role:?} is invalid"))),
            }
        }
        _ => return Err(invalid(format!("head weight role {role:?} is invalid"))),
    };
    Ok(weight)
}

impl PreparedHeadForwardGraphs {
    pub(crate) fn prepare(
        target_device: &Device,
        handles: &AttestedHead,
        load: &ModelLoadPlan,
        geometry: &DecoderGeometry,
        attention: &AttentionGeometry,
        copy: &NativeKernel<copy_rows::Entry>,
        classes: impl IntoIterator<Item = HeadForwardGraphClass>,
    ) -> Result<Self, SubmitError> {
        let handle = handles
            .blocks
            .first()
            .ok_or_else(|| invalid("attested head block is absent"))?;
        let (rotary_components, base) = rotary_components(&attention.rotary)?;
        let p = rotary_components.len() as u64;
        let s = attention
            .width
            .checked_sub(p * 2)
            .ok_or_else(|| invalid("head rotary width exceeds attention width"))?;
        let g = attention
            .heads
            .checked_div(attention.kv_heads)
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid("head grouped-query geometry is invalid"))?;
        let mut variants = Vec::new();
        for class in classes {
            if class.rows == 0 || class.segments == 0 || class.history_rows == 0 {
                return Err(invalid("head forward graph class has a zero extent"));
            }
            if class.segments > class.rows {
                return Err(invalid(
                    "head forward graph has more output segments than rows",
                ));
            }
            if variants
                .iter()
                .any(|variant: &PreparedHeadForwardGraph| variant.class == class)
            {
                return Err(invalid("head forward graph class is duplicated"));
            }
            let mut graph = target_device.native_graph();
            let mut weights = Vec::new();
            macro_rules! weight {
                ($scope:expr, $kind:expr) => {
                    planned_head_weight(
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
            let head_scope = WeightScope::HeadBlock(0);
            let head_dims = [
                ("M", class.rows),
                ("V", geometry.vocabulary),
                ("D", geometry.hidden),
            ];
            let tokens = graph
                .input_for(&handle.input, "tokens", &head_dims)
                .map_err(device)?;
            let copy_dims = [
                ("N", 1),
                ("TS", 1),
                ("TD", class.rows),
                ("KV", 1),
                ("W", geometry.hidden),
            ];
            let mut conditioning_local =
                graph.local_for(copy, "dst", &copy_dims).map_err(device)?;
            let mut conditioning = Vec::with_capacity(class.rows as usize);
            let mut conditioning_from = Vec::with_capacity(class.rows as usize);
            let mut conditioning_to = Vec::with_capacity(class.rows as usize);
            for _ in 0..class.rows {
                let source = graph
                    .port(
                        activation(geometry.activation_dtype),
                        &[1, 1, geometry.hidden],
                    )
                    .map_err(device)?;
                let from = graph.input_for(copy, "from", &copy_dims).map_err(device)?;
                let to = graph.input_for(copy, "to", &copy_dims).map_err(device)?;
                graph
                    .enqueue(
                        copy,
                        copy_rows::WorkflowArgs {
                            src: source.tensor().into(),
                            dst: conditioning_local.tensor_mut().into(),
                            from: from.tensor().into(),
                            to: to.tensor().into(),
                        },
                    )
                    .map_err(device)?;
                conditioning.push(source);
                conditioning_from.push(from);
                conditioning_to.push(to);
            }
            let table = weight!(WeightScope::Target, WeightKind::Embedding);
            let embedding_norm = weight!(head_scope, WeightKind::HeadEmbeddingNorm);
            let hidden_norm = weight!(head_scope, WeightKind::HeadHiddenNorm);
            let combine = weight!(head_scope, WeightKind::HeadCombine);
            let input = graph
                .enqueue(
                    &handle.input,
                    qwen_head_rows::WorkflowArgs {
                        tokens: tokens.tensor().into(),
                        table: (&table).into(),
                        conditioning: conditioning_local.tensor().into(),
                        embedding_norm: (&embedding_norm).into(),
                        hidden_norm: (&hidden_norm).into(),
                        combine: (&combine).into(),
                        epsilon: geometry.epsilon as f32,
                    },
                )
                .map_err(device)?
                .value;
            let input_norm = weight!(head_scope, WeightKind::InputNorm);
            let normalized = graph
                .enqueue(
                    &handle.attention.normalize,
                    qwen_attention_normalize::WorkflowArgs {
                        hidden: (&input).into(),
                        input_norm: (&input_norm).into(),
                        epsilon: geometry.epsilon as f32,
                    },
                )
                .map_err(device)?
                .value;
            let query_norm = weight!(head_scope, WeightKind::QueryNorm);
            let query_gate_weight = weight!(head_scope, WeightKind::QueryGate);
            let key_weight = weight!(head_scope, WeightKind::Key);
            let value_weight = weight!(head_scope, WeightKind::Value);
            let projected = graph
                .enqueue(
                    &handle.attention.project,
                    qwen_attention_project::WorkflowArgs {
                        normalized: (&normalized).into(),
                        query_norm: (&query_norm).into(),
                        query_gate_weight: (&query_gate_weight).into(),
                        key_weight: (&key_weight).into(),
                        value_weight: (&value_weight).into(),
                    },
                )
                .map_err(device)?;
            let prepare_dims = [
                ("M", class.rows),
                ("KV", attention.kv_heads),
                ("G", g),
                ("P", p),
                ("S", s),
            ];
            let coordinates = graph
                .input_for(&handle.attention.prepare, "coordinates", &prepare_dims)
                .map_err(device)?;
            let rotary = graph
                .input_for(
                    &handle.attention.prepare,
                    "rotary_components",
                    &prepare_dims,
                )
                .map_err(device)?;
            let key_norm = weight!(head_scope, WeightKind::KeyNorm);
            let prepared = graph
                .enqueue(
                    &handle.attention.prepare,
                    qwen_attention_prepare::WorkflowArgs {
                        query_gate: (&projected.r0).into(),
                        key: (&projected.r1).into(),
                        query_norm: (&query_norm).into(),
                        key_norm: (&key_norm).into(),
                        coordinates: coordinates.tensor().into(),
                        rotary_components: rotary.tensor().into(),
                        base,
                        epsilon: geometry.epsilon as f32,
                    },
                )
                .map_err(device)?;
            let attend_dims = [
                ("M", class.rows),
                ("T", class.history_rows),
                ("KV", attention.kv_heads),
                ("G", g),
                ("W", attention.width),
                ("R", class.segments),
            ];
            let visible = graph
                .input_for(&handle.attention.attend, "visible", &attend_dims)
                .map_err(device)?;
            let fresh = graph
                .input_for(&handle.attention.attend, "fresh", &attend_dims)
                .map_err(device)?;
            let history_key = graph
                .port(
                    activation(geometry.activation_dtype),
                    &[class.history_rows, attention.kv_heads, attention.width],
                )
                .map_err(device)?;
            let history_value = graph
                .port(
                    activation(geometry.activation_dtype),
                    &[class.history_rows, attention.kv_heads, attention.width],
                )
                .map_err(device)?;
            let mut accumulator = graph
                .local_for(&handle.attention.attend, "accumulator", &attend_dims)
                .map_err(device)?;
            let gated = graph
                .enqueue(
                    &handle.attention.attend,
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
                        scale: 1.0 / (attention.width as f32).sqrt(),
                    },
                )
                .map_err(device)?
                .value;
            let destinations = graph
                .input_for(
                    &handle.attention.output,
                    "destinations",
                    &[
                        ("M", class.rows),
                        ("D", geometry.hidden),
                        ("T", class.history_rows),
                        ("KV", attention.kv_heads),
                        ("G", g),
                        ("W", attention.width),
                    ],
                )
                .map_err(device)?;
            let output_weight = weight!(head_scope, WeightKind::AttentionOutput);
            let mut history_key_mut = history_key.tensor().clone();
            let mut history_value_mut = history_value.tensor().clone();
            let attended = graph
                .enqueue(
                    &handle.attention.output,
                    qwen_attention_output::WorkflowArgs {
                        hidden: (&input).into(),
                        gated: (&gated).into(),
                        prepared_key: (&prepared.r1).into(),
                        value: (&projected.r2).into(),
                        output_weight: (&output_weight).into(),
                        destinations: destinations.tensor().into(),
                        history_key: (&mut history_key_mut).into(),
                        history_value: (&mut history_value_mut).into(),
                    },
                )
                .map_err(device)?
                .value;
            let feedforward_norm = weight!(head_scope, WeightKind::FeedForwardNorm);
            let gate_weight = weight!(head_scope, WeightKind::DenseGate);
            let up_weight = weight!(head_scope, WeightKind::DenseUp);
            let product = graph
                .enqueue(
                    &handle.dense.expand,
                    qwen_dense_expand::WorkflowArgs {
                        residual: (&attended).into(),
                        norm: (&feedforward_norm).into(),
                        gate_weight: (&gate_weight).into(),
                        up_weight: (&up_weight).into(),
                        eps: geometry.epsilon as f32,
                    },
                )
                .map_err(device)?
                .value;
            let down_weight = weight!(head_scope, WeightKind::DenseDown);
            let dense = graph
                .enqueue(
                    &handle.dense.output,
                    qwen_dense_output::WorkflowArgs {
                        residual: (&attended).into(),
                        product: (&product).into(),
                        down_weight: (&down_weight).into(),
                    },
                )
                .map_err(device)?
                .value;
            let feature_dims = [
                ("M", class.rows),
                ("O", class.segments),
                ("D", geometry.hidden),
            ];
            let out_rows = graph
                .input_for(&handle.features, "out_rows", &feature_dims)
                .map_err(device)?;
            let output_norm = weight!(head_scope, WeightKind::OutputNorm);
            let features = graph
                .enqueue(
                    &handle.features,
                    qwen_features_rows::WorkflowArgs {
                        hidden: (&dense).into(),
                        norm: (&output_norm).into(),
                        out_rows: out_rows.tensor().into(),
                        epsilon: geometry.epsilon as f32,
                    },
                )
                .map_err(device)?
                .value;
            graph.export(&features).map_err(device)?;
            let plan = graph.seal().map_err(device)?;
            variants.push(PreparedHeadForwardGraph {
                class,
                plan,
                tokens,
                conditioning,
                conditioning_from,
                conditioning_to,
                coordinates,
                rotary,
                visible,
                fresh,
                destinations,
                history_key,
                history_value,
                out_rows,
                weights,
                features,
            });
        }
        if variants.is_empty() {
            return Err(invalid("head forward graph family has no classes"));
        }
        let plans = variants
            .iter()
            .map(|variant| variant.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self { variants, family })
    }

    pub(crate) fn workspace_bytes_max(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub(crate) fn output_bytes_max(&self) -> u64 {
        self.family.output_bytes()
    }

    pub(crate) fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub(crate) fn plans(&self) -> impl Iterator<Item = (HeadForwardGraphClass, &NativeGraphPlan)> {
        self.variants
            .iter()
            .map(|variant| (variant.class, &variant.plan))
    }

    pub(crate) fn plan(
        &self,
        class: HeadForwardGraphClass,
    ) -> Result<&NativeGraphPlan, SubmitError> {
        Ok(&self.variant(class)?.plan)
    }

    pub(crate) fn run(
        &self,
        class: HeadForwardGraphClass,
        bound: &BoundNativeGraphPlan,
        slot: &mut NativeGraphFamilySlot,
        mut output: NativeGraphOutputLease,
        inputs: HeadForwardGraphInputs<'_>,
    ) -> Result<HeadForwardGraphResult, SubmitError> {
        let variant = self.variant(class)?;
        let mut bindings = bound.bindings();
        if inputs.conditioning.len() != variant.conditioning.len() {
            return Err(invalid(
                "head conditioning row bindings differ from graph class",
            ));
        }
        for (port, tensor) in variant.conditioning.iter().zip(inputs.conditioning) {
            bindings.set(port, tensor).map_err(device)?;
        }
        bindings
            .set(&variant.history_key, inputs.history_key)
            .map_err(device)?;
        bindings
            .set(&variant.history_value, inputs.history_value)
            .map_err(device)?;
        let mut active = slot.activate(&variant.plan).map_err(device)?;
        for (row, (from, to)) in variant
            .conditioning_from
            .iter()
            .zip(&variant.conditioning_to)
            .enumerate()
        {
            active
                .write_input(from, &0_i32.to_le_bytes())
                .map_err(device)?;
            active
                .write_input(to, &(row as i32).to_le_bytes())
                .map_err(device)?;
        }
        for (port, bytes) in [
            (&variant.tokens, inputs.tokens),
            (&variant.coordinates, inputs.coordinates),
            (&variant.rotary, inputs.rotary),
            (&variant.visible, inputs.visible),
            (&variant.fresh, inputs.fresh),
            (&variant.destinations, inputs.destinations),
            (&variant.out_rows, inputs.out_rows),
        ] {
            active.write_input(port, bytes).map_err(device)?;
        }
        let outputs = output
            .activate(&variant.plan)
            .map_err(SubmitError::Invariant)?;
        let outputs = active
            .attach(bindings, outputs)
            .and_then(super::run_graph)
            .map_err(device)?;
        let owner = output.publish(outputs);
        let features = owner
            .tensor(&variant.features)
            .ok_or_else(|| invalid("head forward graph omitted retained features"))?;
        Ok(HeadForwardGraphResult { features })
    }

    fn variant(
        &self,
        class: HeadForwardGraphClass,
    ) -> Result<&PreparedHeadForwardGraph, SubmitError> {
        self.variants
            .iter()
            .find(|variant| variant.class == class)
            .ok_or_else(|| invalid("head forward graph class was not prepared"))
    }
}

pub struct NativeHeadProgram {
    geometry: DecoderGeometry,
    attention: AttentionGeometry,
    graphs: BoundHeadGraphs,
}

impl NativeHeadProgram {
    pub(crate) fn new(
        geometry: DecoderGeometry,
        attention: AttentionGeometry,
        graphs: BoundHeadGraphs,
    ) -> Result<Self, SubmitError> {
        Ok(Self {
            geometry,
            attention,
            graphs,
        })
    }

    fn execute_forward_graph(
        &self,
        core: &HeadLaunchCore,
        graph_workspace: &mut NativeGraphWorkspaceLease,
        graph_output: &mut Option<NativeGraphOutputLease>,
    ) -> Result<GraphOutputTensor, SubmitError> {
        let batch = core.batch().upload();
        let class = batch.class;
        let rows = class.rows();
        let segments = class.segments();
        let mut conditioning = Vec::with_capacity(rows);
        for span in core.conditioning() {
            let source = span.features.allocation().tensor().map_err(device)?;
            for row in span.start..span.start + span.count {
                conditioning.push(
                    source
                        .slice_leading(row as u64, row as u64 + 1)
                        .and_then(|row| row.reshape(&[1, 1, self.geometry.hidden]))
                        .map_err(device)?,
                );
            }
        }
        let padding = conditioning
            .first()
            .cloned()
            .ok_or_else(|| invalid("head conditioning is empty"))?;
        conditioning.resize(rows, padding);
        let mut tokens = vec![0_i32; rows];
        tokens[..batch.class.rows()].copy_from_slice(batch.tokens);
        let mut coordinates = vec![[0_i32; 4]; rows];
        coordinates[..batch.class.rows()].copy_from_slice(batch.coordinates);
        let mut visible = vec![0_i32; rows * segments * 2];
        for (row, ranges) in batch.visible.iter().enumerate() {
            for (range, [start, end]) in ranges.iter().copied().enumerate() {
                let at = (row * segments + range) * 2;
                visible[at] = start;
                visible[at + 1] = end;
            }
        }
        let mut fresh = vec![[0_i32; 2]; rows];
        fresh[..batch.class.rows()].copy_from_slice(batch.fresh);
        let mut destinations = vec![-1_i32; rows];
        destinations[..batch.class.rows()].copy_from_slice(batch.destinations);
        let mut out_rows = vec![0_i32; segments];
        out_rows[..batch.out_rows.len()].copy_from_slice(batch.out_rows);
        let (rotary, _) = rotary_components(&self.attention.rotary)?;
        let history_key = history_plane(core, VectorKind::Key)?;
        let history_value = history_plane(core, VectorKind::Value)?;
        let history_rows = history_key
            .extents()
            .first()
            .copied()
            .ok_or_else(|| invalid("head history has no row axis"))?;
        let graph_class = HeadForwardGraphClass {
            rows: rows as u64,
            segments: segments as u64,
            history_rows,
        };
        let (prepared, bound) = self.graphs.forward(graph_class)?;
        prepared
            .run(
                graph_class,
                bound,
                graph_workspace.slot_mut(),
                graph_output
                    .take()
                    .ok_or_else(|| invalid("head graph output lease is absent"))?,
                HeadForwardGraphInputs {
                    tokens: &i32_bytes(&tokens),
                    conditioning: &conditioning,
                    coordinates: &i32_bytes(
                        &coordinates.iter().flatten().copied().collect::<Vec<_>>(),
                    ),
                    rotary: &i32_bytes(&rotary),
                    visible: &i32_bytes(&visible),
                    fresh: &i32_bytes(&fresh.iter().flatten().copied().collect::<Vec<_>>()),
                    destinations: &i32_bytes(&destinations),
                    history_key: &history_key,
                    history_value: &history_value,
                    out_rows: &i32_bytes(&out_rows),
                },
            )
            .map(|result| result.features)
    }

    fn execute_project_graph(
        &self,
        core: &ProjectLaunchCore,
        graph_workspace: &mut NativeGraphWorkspaceLease,
        graph_output: &mut Option<NativeGraphOutputLease>,
    ) -> Result<ProjectGraphOutput, SubmitError> {
        let rows = core.physical_class().rows();
        let mut features = Vec::with_capacity(rows);
        for request in core.requests() {
            let source = request.features().allocation().tensor().map_err(device)?;
            for row in 0..request.rows() {
                features.push(
                    source
                        .slice_leading(row as u64, row as u64 + 1)
                        .and_then(|row| row.reshape(&[1, 1, self.geometry.hidden]))
                        .map_err(device)?,
                );
            }
        }
        let padding = features
            .first()
            .cloned()
            .ok_or_else(|| invalid("head project features are empty"))?;
        features.resize(rows, padding);
        let vocabulary = self.geometry.vocabulary as usize;
        let mask_words = vocabulary.div_ceil(32);
        let mut shaping = vec![0_f32; rows * 8];
        let mut history = vec![-1_i32; rows * 64];
        let mut masks = vec![0_u32; rows * mask_words];
        let mut draws = vec![0_u32; rows * 6];
        let mut row = 0usize;
        for request in core.requests() {
            let spec = request.select();
            let shape = spec.shaping;
            let shaped = [
                shape.temperature,
                shape.top_k as f32,
                shape.top_p,
                shape.min_p,
                shape.repetition_penalty,
                shape.presence_penalty,
                shape.frequency_penalty,
                0.0,
            ];
            let mut row_history = [-1_i32; 64];
            if let Some(tokens) = &spec.history {
                row_history[..tokens.len()].copy_from_slice(tokens);
            }
            let mut mask = spec
                .mask
                .as_deref()
                .map(Vec::from)
                .unwrap_or_else(|| vec![u32::MAX; mask_words]);
            if vocabulary % 32 != 0 {
                mask[mask_words - 1] &= (1u32 << (vocabulary % 32)) - 1;
            }
            let position = spec.position as u64;
            let draw = [
                match spec.sampling {
                    crate::Sampling::Greedy => 0,
                    crate::Sampling::Categorical => 1,
                },
                spec.seed as u32,
                (spec.seed >> 32) as u32,
                position as u32,
                (position >> 32) as u32,
                spec.domain,
            ];
            for _ in 0..request.rows() {
                shaping[row * 8..(row + 1) * 8].copy_from_slice(&shaped);
                history[row * 64..(row + 1) * 64].copy_from_slice(&row_history);
                masks[row * mask_words..(row + 1) * mask_words].copy_from_slice(&mask);
                draws[row * 6..(row + 1) * 6].copy_from_slice(&draw);
                row += 1;
            }
        }
        let graph_class = HeadProjectGraphClass { rows: rows as u64 };
        let (prepared, bound) = self.graphs.project(graph_class)?;
        let bindings = prepared.bindings(
            graph_class,
            HeadProjectGraphInputs {
                features: &features,
                shaping: &[],
                history: &[],
                mask: &[],
                draws: &[],
            },
            bound,
        )?;
        let result = prepared.run(
            graph_class,
            graph_workspace.slot_mut(),
            bindings,
            graph_output
                .take()
                .ok_or_else(|| invalid("head graph output lease is absent"))?,
            HeadProjectGraphInputs {
                features: &features,
                shaping: &f32_bytes(&shaping),
                history: &i32_bytes(&history),
                mask: &u32_bytes(&masks),
                draws: &u32_bytes(&draws),
            },
        )?;
        Ok(ProjectGraphOutput {
            logits: result.logits,
            selected: result.selected,
        })
    }
}

fn rotary_components(rotary: &RotarySemantics) -> Result<(Vec<i32>, f32), SubmitError> {
    match rotary {
        RotarySemantics::Interleaved {
            width,
            base,
            sections,
            axis_pattern,
        } => {
            let pairs = usize::try_from(width / 2)
                .map_err(|_| invalid("rotary width exceeds host domain"))?;
            if sections.len() < 3 || axis_pattern.len() < 3 {
                return Err(invalid("multimodal rotary requires three coordinate axes"));
            }
            let second_cutoff = usize::try_from(sections[1])
                .map_err(|_| invalid("rotary section exceeds host domain"))?
                .checked_mul(3)
                .ok_or_else(|| invalid("rotary section cutoff overflows"))?;
            let third_cutoff = usize::try_from(sections[2])
                .map_err(|_| invalid("rotary section exceeds host domain"))?
                .checked_mul(3)
                .ok_or_else(|| invalid("rotary section cutoff overflows"))?;
            let components = (0..pairs)
                .map(|index| {
                    let axis = if index % 3 == 1 && index < second_cutoff {
                        1
                    } else if index % 3 == 2 && index < third_cutoff {
                        2
                    } else {
                        0
                    };
                    i32::from(axis_pattern[axis])
                })
                .collect();
            Ok((components, *base as f32))
        }
    }
}

#[cfg(test)]
mod rotary_tests {
    use super::*;

    #[test]
    fn multimodal_rotary_interleaves_axes_with_section_cutoffs() {
        let rotary = RotarySemantics::Interleaved {
            width: 12,
            base: 10_000.0,
            sections: vec![2, 2, 1, 0],
            axis_pattern: vec![0, 1, 2],
        };
        let (components, _) = rotary_components(&rotary).expect("valid rotary");
        assert_eq!(components, vec![0, 1, 2, 0, 1, 0]);
    }
}

fn history_plane(core: &HeadLaunchCore, vector: VectorKind) -> Result<Tensor, SubmitError> {
    for advance in core.advances() {
        let bindings = advance.bindings();
        if let Some(plane) = bindings.history.iter().find(|plane| {
            plane.layer == LayerRef::Head(0)
                && plane.vector == vector
                && plane.name == PlaneName::Dense
        }) {
            return Ok(plane.buffer.clone());
        }
    }
    Err(invalid("head history plane is absent"))
}

impl HeadProgram for NativeHeadProgram {
    type Submission = ReadySubmission<HeadLaunchCore, NativeGraphWorkspaceLease, GraphOutputTensor>;
    type ProjectSubmission =
        ReadySubmission<ProjectLaunchCore, NativeGraphWorkspaceLease, ProjectGraphOutput>;

    fn submit(
        &mut self,
        mut launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)> {
        let result = {
            let (core, graph_workspace, graph_output) = launch.execution_parts_mut();
            self.execute_forward_graph(core, graph_workspace, graph_output)
        };
        let features = match result {
            Ok(features) => features,
            Err(error) => return Err((error, launch)),
        };
        let (core, graph_workspace, _) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, graph_workspace, features))
    }

    fn project(
        &mut self,
        mut launch: ValidatedProjectionLaunch,
    ) -> Result<Self::ProjectSubmission, (SubmitError, ValidatedProjectionLaunch)> {
        let result = {
            let (core, graph_workspace, graph_output) = launch.execution_parts_mut();
            self.execute_project_graph(core, graph_workspace, graph_output)
        };
        let output = match result {
            Ok(output) => output,
            Err(error) => return Err((error, launch)),
        };
        let (core, graph_workspace, _) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, graph_workspace, output))
    }
}
