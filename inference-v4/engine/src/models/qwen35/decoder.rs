//! Qwen sequence execution through generated Seismic bindings. Runtime tensor
//! descriptors are the complete specialization domain; the engine owns no
//! compiler plans, shape envelopes, capacity classes, or raw backend buffers.

mod conditioning;

use super::{Description, FeedForwardWeights, Geometry, HeadMapping, MixerWeights};
use crate::{
    generation::{
        sampling::{Sampler, Selection},
        Proposal, Sampling,
    },
    kernels,
    models::sequence::{Advance, OwnedSequence, SequenceWork},
    state::{AdvanceBindings, ComponentSpec, SequenceState, StateAdvance, StateStore},
    weights::{descriptor::WeightDescriptor, residency::ResidentWeight},
    Error,
};
use conditioning::{Conditioning, Overlay};
use seismic::{DType, Device, Element, Kernel, PrecisionPolicy, Tensor, Workflow};
use std::{
    collections::{hash_map::Entry as HashEntry, HashMap},
    rc::Rc,
    time::Instant,
};

/// Model-state capacity, not a compiler specialization envelope.
pub struct DecoderCapacity {
    pub context_capacity: usize,
    pub max_sequences: usize,
}

struct Embedding {
    kernel: Kernel<kernels::qwen_embedding_rows::Entry>,
    table: ResidentWeight,
}

struct Attention {
    kernel: Kernel<kernels::qwen_attention_sequence::Entry>,
    input_norm: ResidentWeight,
    query_gate: ResidentWeight,
    key: ResidentWeight,
    value: ResidentWeight,
    query_norm: ResidentWeight,
    key_norm: ResidentWeight,
    output: ResidentWeight,
    rotary_components: Tensor,
    base: f32,
    epsilon: f32,
    scale: f32,
}

struct Recurrent {
    kernel: Kernel<kernels::qwen_recurrent_sequence::Entry>,
    input_norm: ResidentWeight,
    qkv: ResidentWeight,
    gate: ResidentWeight,
    alpha: ResidentWeight,
    beta: ResidentWeight,
    convolution: ResidentWeight,
    rate: ResidentWeight,
    time_bias: ResidentWeight,
    norm: ResidentWeight,
    output: ResidentWeight,
    epsilon: f32,
    preparation_epsilon: f32,
    grouped: bool,
}

enum Mixer {
    Attention(Attention),
    Recurrent(Recurrent),
}

struct Dense {
    kernel: Kernel<kernels::qwen_dense_suffix::Entry>,
    norm: ResidentWeight,
    gate: ResidentWeight,
    up: ResidentWeight,
    down: ResidentWeight,
    epsilon: f32,
}

struct Routed {
    kernel: Kernel<kernels::qwen_routed_suffix::Entry>,
    selected: u64,
    norm: ResidentWeight,
    router: ResidentWeight,
    shared_router: ResidentWeight,
    expert_gate: ResidentWeight,
    expert_up: ResidentWeight,
    expert_down: ResidentWeight,
    shared_gate: ResidentWeight,
    shared_up: ResidentWeight,
    shared_down: ResidentWeight,
    epsilon: f32,
    normalize: bool,
}

enum FeedForward {
    Dense(Dense),
    Routed(Routed),
}

struct Block {
    mixer: Mixer,
    feedforward: FeedForward,
    state_index: usize,
}

struct ReadoutKernels {
    rows: Kernel<kernels::qwen_readout_rows::Entry>,
    selected: Kernel<kernels::qwen_readout_selected::Entry>,
    norm: ResidentWeight,
    weight: ResidentWeight,
    epsilon: f32,
}

struct Rows {
    logits: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    tokens: Tensor,
    destinations: Tensor,
}

struct SelectedRows {
    ids: Tensor,
    logits: Tensor,
}

enum PendingReadout {
    Rows(kernels::qwen_readout_rows::WorkflowResults),
    Selected(kernels::qwen_readout_selected::WorkflowResults),
}

pub struct Decoder {
    geometry: Geometry,
    store: Rc<StateStore>,
    device: Rc<Device>,
    conditioning: Conditioning,
    context_capacity: usize,
    embedding: Embedding,
    blocks: Vec<Block>,
    readout: ReadoutKernels,
    sampler: Sampler,
    rows: HashMap<(usize, usize), Rows>,
    selected_rows: HashMap<usize, SelectedRows>,
}

#[derive(Clone, Debug)]
pub struct StageExecutionObservation {
    pub host_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct DecoderStepObservation {
    pub stage: String,
    pub block: Option<usize>,
    pub entry: String,
    pub execution: StageExecutionObservation,
}

#[derive(Clone, Debug)]
pub struct DecoderBatchObservation {
    pub host_seconds: f64,
}

struct ConditionedInput<'a> {
    coordinates: &'a [[i32; 4]],
    overlays: &'a [Overlay],
}

fn element(dtype: DType) -> Element {
    Element::dense(dtype)
}

fn shape(value: usize, what: &str) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| format!("{what} exceeds the Seismic shape domain").into())
}

fn tensor_bytes(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>()
}

fn rows_entry<'a>(
    geometry: &Geometry,
    device: &Device,
    rows: &'a mut HashMap<(usize, usize), Rows>,
    count: usize,
    ranges: usize,
    context_capacity: usize,
) -> Result<&'a mut Rows, Error> {
    if count == 0 || ranges == 0 || count > context_capacity {
        return Err("invalid forward row count".into());
    }
    match rows.entry((count, ranges)) {
        HashEntry::Occupied(entry) => Ok(entry.into_mut()),
        HashEntry::Vacant(entry) => {
            let m = shape(count, "forward row count")?;
            let r = shape(ranges, "visibility range count")?;
            Ok(entry.insert(Rows {
                logits: Tensor::zeros(device, Element::f32(), &[1, geometry.vocabulary])?,
                coordinates: Tensor::zeros(device, Element::i32(), &[m, 4])?,
                visible: Tensor::zeros(device, Element::i32(), &[m, r, 2])?,
                tokens: Tensor::zeros(device, Element::i32(), &[m])?,
                destinations: Tensor::zeros(device, Element::i32(), &[m])?,
            }))
        }
    }
}

fn selected_entry<'a>(
    device: &Device,
    selected: &'a mut HashMap<usize, SelectedRows>,
    ids: usize,
) -> Result<&'a mut SelectedRows, Error> {
    if ids == 0 {
        return Err("selected readout must contain at least one id".into());
    }
    match selected.entry(ids) {
        HashEntry::Occupied(entry) => Ok(entry.into_mut()),
        HashEntry::Vacant(entry) => {
            let extent = shape(ids, "selected readout size")?;
            Ok(entry.insert(SelectedRows {
                ids: Tensor::zeros(device, Element::i32(), &[extent])?,
                logits: Tensor::zeros(device, Element::f32(), &[1, extent])?,
            }))
        }
    }
}

fn observed<T>(
    observations: &mut Option<Vec<DecoderStepObservation>>,
    stage: &str,
    block: Option<usize>,
    entry: &str,
    call: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    let started = Instant::now();
    let result = call()?;
    if let Some(observations) = observations {
        observations.push(DecoderStepObservation {
            stage: stage.into(),
            block,
            entry: entry.into(),
            execution: StageExecutionObservation {
                host_seconds: started.elapsed().as_secs_f64(),
            },
        });
    }
    Ok(result)
}

impl Embedding {
    fn execute(&self, tokens: &Tensor) -> Result<Tensor, Error> {
        Ok(self
            .kernel
            .call(kernels::qwen_embedding_rows::Args {
                table: self.table.tensor(),
                tokens,
            })?
            .value)
    }
}

impl Block {
    fn attention(&self) -> bool {
        matches!(self.mixer, Mixer::Attention(_))
    }

    fn mix(
        &self,
        hidden: &Tensor,
        rows: &Rows,
        transition: AdvanceBindings<'_>,
    ) -> Result<(Tensor, Vec<(usize, Tensor)>), Error> {
        match &self.mixer {
            Mixer::Attention(mixer) => {
                let index = self.state_index;
                let mut history_key = transition.history[index].clone();
                let mut history_value = transition.history[index + 1].clone();
                let result = mixer.kernel.call(kernels::qwen_attention_sequence::Args {
                    hidden,
                    input_norm: mixer.input_norm.tensor(),
                    query_gate_weight: mixer.query_gate.tensor(),
                    key_weight: mixer.key.tensor(),
                    value_weight: mixer.value.tensor(),
                    query_norm: mixer.query_norm.tensor(),
                    key_norm: mixer.key_norm.tensor(),
                    output_weight: mixer.output.tensor(),
                    coordinates: &rows.coordinates,
                    rotary_components: &mixer.rotary_components,
                    visible: &rows.visible,
                    history_key: &mut history_key,
                    history_value: &mut history_value,
                    destinations: &rows.destinations,
                    base: mixer.base,
                    epsilon: mixer.epsilon,
                    scale: mixer.scale,
                })?;
                Ok((result.value, Vec::new()))
            }
            Mixer::Recurrent(mixer) => {
                let index = self.state_index;
                let result = mixer.kernel.call(kernels::qwen_recurrent_sequence::Args {
                    hidden,
                    input_norm: mixer.input_norm.tensor(),
                    qkv_weight: mixer.qkv.tensor(),
                    gate_weight: mixer.gate.tensor(),
                    alpha_weight: mixer.alpha.tensor(),
                    beta_weight: mixer.beta.tensor(),
                    convolution: mixer.convolution.tensor(),
                    rate: mixer.rate.tensor(),
                    time_bias: mixer.time_bias.tensor(),
                    recurrent_norm: mixer.norm.tensor(),
                    output_weight: mixer.output.tensor(),
                    window: &transition.previous[index],
                    delta: &transition.previous[index + 1],
                    epsilon: mixer.epsilon,
                    preparation_epsilon: mixer.preparation_epsilon,
                    grouped: mixer.grouped,
                })?;
                Ok((result.r2, vec![(index, result.r0), (index + 1, result.r1)]))
            }
        }
    }

    fn feedforward(&self, residual: &Tensor) -> Result<Tensor, Error> {
        match &self.feedforward {
            FeedForward::Dense(feedforward) => Ok(feedforward
                .kernel
                .call(kernels::qwen_dense_suffix::Args {
                    residual,
                    norm: feedforward.norm.tensor(),
                    gate_weight: feedforward.gate.tensor(),
                    up_weight: feedforward.up.tensor(),
                    down_weight: feedforward.down.tensor(),
                    eps: feedforward.epsilon,
                })?
                .r6),
            FeedForward::Routed(feedforward) => {
                let rows = *residual
                    .extents()
                    .first()
                    .ok_or("routed residual tensor has no row axis")?;
                let mut routes = Tensor::zeros(
                    &residual.device(),
                    Element::i32(),
                    &[rows, feedforward.selected],
                )?;
                let mut scores = Tensor::zeros(
                    &residual.device(),
                    Element::f32(),
                    &[rows, feedforward.selected],
                )?;
                Ok(feedforward
                    .kernel
                    .call(kernels::qwen_routed_suffix::Args {
                        residual,
                        norm: feedforward.norm.tensor(),
                        router: feedforward.router.tensor(),
                        shared_router: feedforward.shared_router.tensor(),
                        expert_gate: feedforward.expert_gate.tensor(),
                        expert_up: feedforward.expert_up.tensor(),
                        expert_down: feedforward.expert_down.tensor(),
                        shared_gate: feedforward.shared_gate.tensor(),
                        shared_up: feedforward.shared_up.tensor(),
                        shared_down: feedforward.shared_down.tensor(),
                        routes: &mut routes,
                        scores: &mut scores,
                        eps: feedforward.epsilon,
                        normalize: i32::from(feedforward.normalize),
                    })?
                    .r14)
            }
        }
    }
}

/// Explicit numerical output; state-only execution omits vocabulary projection.
pub enum Readout<'a> {
    StateOnly,
    Logits,
    Selected(&'a [u32]),
    Sample {
        mask: Option<&'a [u32]>,
        sampling: Sampling,
        seed: u64,
        position: usize,
    },
}

#[derive(Debug)]
pub enum ReadoutOutput {
    StateOnly,
    Logits(Vec<f32>),
    Selected(Vec<f32>),
    Sample(Selection),
}

fn generation_selection(output: &ReadoutOutput) -> Result<Option<crate::inputs::TokenId>, String> {
    match output {
        ReadoutOutput::Sample(Selection::Token(token)) => Ok(Some(*token)),
        ReadoutOutput::Sample(Selection::Empty) => Err("empty sampling distribution".into()),
        ReadoutOutput::Sample(Selection::Nonfinite) => {
            Err("nonfinite sampling distribution".into())
        }
        ReadoutOutput::StateOnly => Ok(None),
        ReadoutOutput::Logits(_) | ReadoutOutput::Selected(_) => {
            unreachable!("generation never requests host logits")
        }
    }
}

pub struct ExecutedAdvance<'a> {
    advance: StateAdvance<'a>,
    output: ReadoutOutput,
}
impl ExecutedAdvance<'_> {
    pub fn output(&self) -> &ReadoutOutput {
        &self.output
    }
    pub fn commit(self) -> Result<(), String> {
        self.advance.commit()
    }
    pub fn abort(self) {
        self.advance.abort();
    }
}

pub struct ConditionedAdvance<'a> {
    advance: ExecutedAdvance<'a>,
    input: &'a mut super::inputs::InputState,
    next: super::inputs::InputState,
}
impl ConditionedAdvance<'_> {
    pub fn output(&self) -> &ReadoutOutput {
        self.advance.output()
    }
    pub fn commit(self) -> Result<(), String> {
        self.advance.commit()?;
        *self.input = self.next;
        Ok(())
    }
    pub fn abort(self) {
        self.advance.abort();
    }
}

pub struct DecodedAdvance<'a> {
    advance: StateAdvance<'a>,
    logits: Vec<f32>,
}
impl<'a> ExecutedAdvance<'a> {
    fn decoded(self) -> DecodedAdvance<'a> {
        let ReadoutOutput::Logits(logits) = self.output else {
            unreachable!("logit API requested logit readout")
        };
        DecodedAdvance {
            advance: self.advance,
            logits,
        }
    }
}
impl DecodedAdvance<'_> {
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn commit(self) -> Result<(), String> {
        self.advance.commit()
    }
    pub fn abort(self) {
        self.advance.abort();
    }
}

pub struct GenerationWork<'a, S = ()> {
    pub sequence: &'a OwnedSequence<S>,
    pub proposal: &'a Proposal,
    pub mask: Option<&'a [u32]>,
}

impl Decoder {
    pub fn compile(
        device: Rc<Device>,
        description: &Description,
        mut import: impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, Error>,
        precision: PrecisionPolicy,
        capacity: DecoderCapacity,
    ) -> Result<Self, Error> {
        let geometry = &description.geometry;
        geometry.validate().map_err(|error| error.to_string())?;
        if description.blocks.len() != geometry.layers.len() {
            return Err("decoder requires complete block descriptors".into());
        }
        let DecoderCapacity {
            context_capacity,
            max_sequences,
        } = capacity;
        if context_capacity == 0
            || context_capacity as u64 > geometry.context_limit
            || context_capacity > i32::MAX as usize
            || max_sequences == 0
        {
            return Err("invalid decoder state capacity".into());
        }
        let history_capacity = context_capacity
            .checked_mul(max_sequences)
            .filter(|value| *value <= i32::MAX as usize)
            .ok_or("history capacity overflow")?;
        let activation = geometry.activation_dtype;
        let activation_element = element(activation);
        let rotary_base = geometry.rotary_base as f32;
        let epsilon = geometry.epsilon as f32;
        let preparation_epsilon = (geometry.epsilon * geometry.recurrent_width as f64) as f32;
        let attention_scale = 1.0 / (geometry.attention_width as f32).sqrt();
        if !rotary_base.is_finite()
            || rotary_base <= 0.0
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || !preparation_epsilon.is_finite()
            || preparation_epsilon <= 0.0
            || !attention_scale.is_finite()
            || attention_scale <= 0.0
        {
            return Err("Qwen numerical parameters are not representable as f32".into());
        }

        // Rotary coordinate selection is model semantics. Materialize its
        // immutable per-pair map once; the tensor axis also exposes P to the
        // generated call schema, allowing S to be inferred exactly from 2P+S.
        let rotary_pairs = geometry.rotary_width / 2;
        let height_limit = geometry.rotary_sections[1]
            .checked_mul(3)
            .ok_or("rotary height section overflow")?;
        let width_limit = geometry.rotary_sections[2]
            .checked_mul(3)
            .ok_or("rotary width section overflow")?;
        let rotary_components = Tensor::from_host(
            &device,
            Element::i32(),
            &[rotary_pairs],
            &tensor_bytes((0..rotary_pairs).map(|pair| {
                if pair % 3 == 1 && pair < height_limit {
                    1
                } else if pair % 3 == 2 && pair < width_limit {
                    2
                } else {
                    0
                }
            })),
        )?;

        let mut import_checked =
            |descriptor: &WeightDescriptor, dtype: DType| -> Result<ResidentWeight, Error> {
                let weight = import(descriptor, dtype)?;
                if !weight.belongs_to(&device) {
                    return Err(Error::from(format!(
                        "imported weight {} belongs to another device",
                        descriptor.name
                    )));
                }
                Ok(weight)
            };

        let table = import_checked(&description.embedding, activation)?;
        let embedding = Embedding {
            kernel: kernels::qwen_embedding_rows::for_device_with(
                &device,
                precision.clone(),
                kernels::qwen_embedding_rows::Elements {
                    EW: table.element(),
                    A: activation_element,
                },
            )?,
            table,
        };

        let mut blocks = Vec::with_capacity(description.blocks.len());
        let mut components = Vec::new();
        let mut history_rows = Vec::new();
        for (index, block) in description.blocks.iter().enumerate() {
            let input_norm = import_checked(&block.input_norm, activation)?;
            let (mixer, state_index) = match &block.mixer {
                MixerWeights::Attention(weights) => {
                    let state_index = history_rows.len();
                    let spec = ComponentSpec {
                        shape: vec![
                            usize::try_from(geometry.kv_heads)
                                .map_err(|_| "KV heads exceed host range")?,
                            usize::try_from(geometry.attention_width)
                                .map_err(|_| "attention width exceeds host range")?,
                        ],
                        dtype: activation,
                    };
                    history_rows.extend([spec.clone(), spec]);
                    let query_gate = import_checked(&weights.query_gate, activation)?;
                    let key = import_checked(&weights.key, activation)?;
                    let value = import_checked(&weights.value, activation)?;
                    let query_norm = import_checked(&weights.query_norm, DType::F32)?;
                    let key_norm = import_checked(&weights.key_norm, DType::F32)?;
                    let output = import_checked(&weights.output, activation)?;
                    let kernel = kernels::qwen_attention_sequence::for_device_with(
                        &device,
                        precision.clone(),
                        kernels::qwen_attention_sequence::Elements {
                            NW: input_norm.element(),
                            QW: query_gate.element(),
                            KW: key.element(),
                            VW: value.element(),
                            OW: output.element(),
                            A: activation_element,
                        },
                    )?;
                    (
                        Mixer::Attention(Attention {
                            kernel,
                            input_norm,
                            query_gate,
                            key,
                            value,
                            query_norm,
                            key_norm,
                            output,
                            rotary_components: rotary_components.clone(),
                            base: rotary_base,
                            epsilon,
                            scale: attention_scale,
                        }),
                        state_index,
                    )
                }
                MixerWeights::Recurrent(weights) => {
                    let state_index = components.len();
                    let host = |value| {
                        usize::try_from(value).map_err(|_| "component exceeds address range")
                    };
                    components.push(ComponentSpec {
                        shape: vec![
                            host(geometry.convolution_width - 1)?,
                            host(geometry.recurrent_channels().map_err(|e| e.to_string())?)?,
                        ],
                        dtype: activation,
                    });
                    components.push(ComponentSpec {
                        shape: vec![
                            host(geometry.recurrent_value_heads)?,
                            host(geometry.recurrent_width)?,
                            host(geometry.recurrent_width)?,
                        ],
                        dtype: DType::F32,
                    });
                    let qkv = import_checked(&weights.query_key_value, activation)?;
                    let gate = import_checked(&weights.gate, activation)?;
                    let alpha = import_checked(&weights.alpha, activation)?;
                    let beta = import_checked(&weights.beta, activation)?;
                    let convolution = import_checked(&weights.convolution, DType::F32)?;
                    let rate = import_checked(&weights.decay, DType::F32)?;
                    let time_bias = import_checked(&weights.time_bias, DType::F32)?;
                    let norm = import_checked(&weights.norm, activation)?;
                    let output = import_checked(&weights.output, activation)?;
                    let kernel = kernels::qwen_recurrent_sequence::for_device_with(
                        &device,
                        precision.clone(),
                        kernels::qwen_recurrent_sequence::Elements {
                            NW: input_norm.element(),
                            QW: qkv.element(),
                            GW: gate.element(),
                            AW: alpha.element(),
                            BW: beta.element(),
                            RN: norm.element(),
                            OW: output.element(),
                            A: activation_element,
                        },
                    )?;
                    (
                        Mixer::Recurrent(Recurrent {
                            kernel,
                            input_norm,
                            qkv,
                            gate,
                            alpha,
                            beta,
                            convolution,
                            rate,
                            time_bias,
                            norm,
                            output,
                            epsilon,
                            preparation_epsilon,
                            grouped: geometry.recurrent_head_mapping == HeadMapping::Grouped,
                        }),
                        state_index,
                    )
                }
            };
            if (geometry.layers[index] == super::MixerKind::Attention)
                != matches!(mixer, Mixer::Attention(_))
            {
                return Err("block descriptor disagrees with layer order".into());
            }

            let feedforward = match &block.feedforward {
                FeedForwardWeights::Dense(weights) => {
                    if geometry.experts.is_some() {
                        return Err("dense feedforward disagrees with routed geometry".into());
                    }
                    let norm = import_checked(&block.feedforward_norm, activation)?;
                    let gate = import_checked(&weights.gate, activation)?;
                    let up = import_checked(&weights.up, activation)?;
                    let down = import_checked(&weights.down, activation)?;
                    let kernel = kernels::qwen_dense_suffix::for_device_with(
                        &device,
                        precision.clone(),
                        kernels::qwen_dense_suffix::Elements {
                            A: activation_element,
                            NW: norm.element(),
                            GW: gate.element(),
                            UW: up.element(),
                            DW: down.element(),
                        },
                    )?;
                    FeedForward::Dense(Dense {
                        kernel,
                        norm,
                        gate,
                        up,
                        down,
                        epsilon,
                    })
                }
                FeedForwardWeights::Routed(weights) => {
                    let experts = geometry
                        .experts
                        .as_ref()
                        .ok_or("routed feedforward lacks expert geometry")?;
                    let norm = import_checked(&block.feedforward_norm, activation)?;
                    let router = import_checked(&weights.router, activation)?;
                    let shared_router = import_checked(&weights.shared_router, DType::F32)?;
                    let expert_gate = import_checked(&weights.expert_gate, activation)?;
                    let expert_up = import_checked(&weights.expert_up, activation)?;
                    let expert_down = import_checked(&weights.expert_down, activation)?;
                    let shared_gate = import_checked(&weights.shared_gate, activation)?;
                    let shared_up = import_checked(&weights.shared_up, activation)?;
                    let shared_down = import_checked(&weights.shared_down, activation)?;
                    let kernel = kernels::qwen_routed_suffix::for_device_with(
                        &device,
                        precision.clone(),
                        kernels::qwen_routed_suffix::Elements {
                            A: activation_element,
                            NW: norm.element(),
                            RW: router.element(),
                            SRW: shared_router.element(),
                            EGW: expert_gate.element(),
                            EUW: expert_up.element(),
                            EDW: expert_down.element(),
                            SGW: shared_gate.element(),
                            SUW: shared_up.element(),
                            SDW: shared_down.element(),
                        },
                    )?;
                    FeedForward::Routed(Routed {
                        kernel,
                        selected: experts.selected,
                        norm,
                        router,
                        shared_router,
                        expert_gate,
                        expert_up,
                        expert_down,
                        shared_gate,
                        shared_up,
                        shared_down,
                        epsilon,
                        normalize: experts.normalize_selected,
                    })
                }
            };
            blocks.push(Block {
                mixer,
                feedforward,
                state_index,
            });
        }

        let output_norm = import_checked(&description.output_norm, activation)?;
        let output = import_checked(&description.output, activation)?;
        let readout_elements = || kernels::qwen_readout_rows::Elements {
            A: activation_element,
            NW: output_norm.element(),
            OW: output.element(),
        };
        let readout = ReadoutKernels {
            rows: kernels::qwen_readout_rows::for_device_with(
                &device,
                precision.clone(),
                readout_elements(),
            )?,
            selected: kernels::qwen_readout_selected::for_device_with(
                &device,
                precision.clone(),
                kernels::qwen_readout_selected::Elements {
                    A: activation_element,
                    NW: output_norm.element(),
                    OW: output.element(),
                },
            )?,
            norm: output_norm,
            weight: output,
            epsilon,
        };
        let store = StateStore::new(
            device.clone(),
            context_capacity,
            history_capacity,
            history_rows,
            components,
        )?;
        let vocabulary = usize::try_from(geometry.vocabulary)
            .map_err(|_| "vocabulary exceeds the host index domain")?;
        let sampler = Sampler::compile(&device, vocabulary, precision.clone())?;
        let conditioning = Conditioning::new(&device, precision, geometry.hidden)?;
        let mut rows = HashMap::new();
        rows_entry(geometry, &device, &mut rows, 1, 1, context_capacity)?;
        Ok(Self {
            geometry: geometry.clone(),
            store,
            device,
            conditioning,
            context_capacity,
            embedding,
            blocks,
            readout,
            sampler,
            rows,
            selected_rows: HashMap::new(),
        })
    }

    fn rows_for(&mut self, count: usize, ranges: usize) -> Result<&mut Rows, Error> {
        rows_entry(
            &self.geometry,
            &self.device,
            &mut self.rows,
            count,
            ranges,
            self.context_capacity,
        )
    }
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    pub fn context_capacity(&self) -> usize {
        self.context_capacity
    }
    pub fn memory_usage(&self) -> seismic::MemoryUsage {
        self.device.memory_usage()
    }
    pub fn reclaim_idle(&mut self) -> Result<usize, Error> {
        let before = self.device.memory_usage().charged;
        self.rows.retain(|geometry, _| *geometry == (1, 1));
        self.selected_rows.clear();
        self.store.release_idle()?;
        let released = before.saturating_sub(self.device.memory_usage().charged);
        usize::try_from(released).map_err(|_| "released byte count exceeds host range".into())
    }
    pub fn state_store(&self) -> &Rc<StateStore> {
        &self.store
    }
    pub fn compiled_kernel_count(&self) -> usize {
        5 + self.blocks.len() * 2
    }

    pub fn prepare_generation_batch(
        &mut self,
        work: &[GenerationWork<'_>],
    ) -> Result<Vec<Box<dyn Advance>>, Error> {
        let sequences = work
            .iter()
            .map(|row| SequenceWork {
                sequence: row.sequence,
                position: row.proposal.position(),
                count: row.proposal.tokens().len(),
            })
            .collect::<Vec<_>>();
        OwnedSequence::prepare_completed_batch(&sequences, |states| {
            let mut selected = Vec::with_capacity(work.len());
            for (row, state) in work.iter().zip(states.iter_mut()) {
                let tokens = row
                    .proposal
                    .tokens()
                    .iter()
                    .map(|token| token.0)
                    .collect::<Vec<_>>();
                let readout = if row.proposal.needs_sample() {
                    Readout::Sample {
                        mask: row.mask,
                        sampling: row.proposal.sampling(),
                        seed: row.proposal.seed(),
                        position: row.proposal.sample_position(),
                    }
                } else {
                    Readout::StateOnly
                };
                let advance = self.execute(state, &tokens, readout)?;
                let outcome = generation_selection(advance.output());
                advance.commit()?;
                selected.push(outcome);
            }
            Ok(selected)
        })
    }

    pub fn prepare_conditioned_generation_batch(
        &mut self,
        work: &[GenerationWork<'_, super::inputs::InputState>],
    ) -> Result<Vec<Box<dyn Advance>>, Error> {
        let sequences = work
            .iter()
            .map(|row| SequenceWork {
                sequence: row.sequence,
                position: row.proposal.position(),
                count: row.proposal.tokens().len(),
            })
            .collect::<Vec<_>>();
        OwnedSequence::prepare_completed_batch_with_semantics(&sequences, |states| {
            let mut selected = Vec::with_capacity(work.len());
            for (row, (state, input)) in work.iter().zip(states.iter_mut()) {
                let tokens = row
                    .proposal
                    .tokens()
                    .iter()
                    .map(|token| token.0)
                    .collect::<Vec<_>>();
                let readout = if row.proposal.needs_sample() {
                    Readout::Sample {
                        mask: row.mask,
                        sampling: row.proposal.sampling(),
                        seed: row.proposal.seed(),
                        position: row.proposal.sample_position(),
                    }
                } else {
                    Readout::StateOnly
                };
                let advance = self.execute_conditioned(state, input, &tokens, readout)?;
                let outcome = generation_selection(advance.output());
                advance.commit()?;
                selected.push(outcome);
            }
            Ok(selected)
        })
    }

    pub fn propose<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<DecodedAdvance<'a>, Error> {
        self.propose_impl(state, &[token], false, Readout::Logits, None)
            .map(|(advance, _)| advance.decoded())
    }
    pub fn propose_observed<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<(DecodedAdvance<'a>, Vec<DecoderStepObservation>), Error> {
        self.propose_impl(state, &[token], true, Readout::Logits, None)
            .map(|(advance, steps)| (advance.decoded(), steps))
    }
    pub fn propose_batched<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<(DecodedAdvance<'a>, DecoderBatchObservation), Error> {
        let started = Instant::now();
        let advance = self.propose(state, token)?;
        Ok((
            advance,
            DecoderBatchObservation {
                host_seconds: started.elapsed().as_secs_f64(),
            },
        ))
    }
    pub fn prefill<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
    ) -> Result<DecodedAdvance<'a>, Error> {
        self.propose_impl(state, tokens, false, Readout::Logits, None)
            .map(|(advance, _)| advance.decoded())
    }
    pub fn prefill_observed<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
    ) -> Result<(DecodedAdvance<'a>, Vec<DecoderStepObservation>), Error> {
        self.propose_impl(state, tokens, true, Readout::Logits, None)
            .map(|(advance, steps)| (advance.decoded(), steps))
    }
    pub fn prefill_batched<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
    ) -> Result<(DecodedAdvance<'a>, DecoderBatchObservation), Error> {
        let started = Instant::now();
        let advance = self.prefill(state, tokens)?;
        Ok((
            advance,
            DecoderBatchObservation {
                host_seconds: started.elapsed().as_secs_f64(),
            },
        ))
    }
    pub fn execute<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
        readout: Readout<'_>,
    ) -> Result<ExecutedAdvance<'a>, Error> {
        self.propose_impl(state, tokens, false, readout, None)
            .map(|(advance, _)| advance)
    }
    pub fn execute_conditioned<'a>(
        &mut self,
        state: &'a mut SequenceState,
        input: &'a mut super::inputs::InputState,
        tokens: &[u32],
        readout: Readout<'_>,
    ) -> Result<ConditionedAdvance<'a>, Error> {
        if input.position() != state.position()
            || input.width() as u64 != self.geometry.hidden
            || !input.belongs_to(&self.device)
        {
            return Err("conditioned input differs from decoder continuation or owner".into());
        }
        let assembled = input.assemble(tokens)?;
        let overlays = self.conditioning.prepare(&assembled)?;
        let next = input.after(
            input
                .position()
                .checked_add(tokens.len())
                .ok_or("input advance overflow")?,
        )?;
        let coordinates = assembled
            .coordinates
            .iter()
            .map(|&[t, h, w]| [t, h, w, 0])
            .collect::<Vec<_>>();
        let conditioned = ConditionedInput {
            coordinates: &coordinates,
            overlays: &overlays,
        };
        let (advance, _) = self.propose_impl(state, tokens, false, readout, Some(&conditioned))?;
        Ok(ConditionedAdvance {
            advance,
            input,
            next,
        })
    }

    fn propose_impl<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
        collect_observations: bool,
        readout: Readout<'_>,
        conditioned: Option<&ConditionedInput<'_>>,
    ) -> Result<(ExecutedAdvance<'a>, Vec<DecoderStepObservation>), Error> {
        if !state.belongs_to(&self.store) {
            return Err("sequence belongs to another decoder state store".into());
        }
        if tokens.is_empty()
            || tokens.iter().any(|&token| {
                u64::from(token) >= self.geometry.vocabulary || token > i32::MAX as u32
            })
        {
            return Err("token is outside vocabulary".into());
        }
        if tokens.len() > self.context_capacity - state.position() {
            return Err("forward exceeds context limit".into());
        }
        let Decoder {
            geometry,
            device,
            conditioning,
            context_capacity,
            embedding,
            blocks,
            readout: readout_kernels,
            sampler,
            rows: row_cache,
            selected_rows,
            ..
        } = self;
        let history_ranges = state.history_ranges();
        let range_count = history_ranges.len().max(1);
        let selected_ids = match &readout {
            Readout::Selected(ids) => {
                if ids.iter().any(|&id| u64::from(id) >= geometry.vocabulary) {
                    return Err("selected readout token is outside vocabulary".into());
                }
                Some(*ids)
            }
            _ => None,
        };
        let mut selected = selected_ids
            .filter(|ids| !ids.is_empty())
            .map(|ids| {
                let selected = selected_entry(device, selected_rows, ids.len())?;
                selected
                    .ids
                    .write_from_host(&tensor_bytes(ids.iter().map(|&id| id as i32)))?;
                Ok::<_, Error>(selected)
            })
            .transpose()?;
        let rows = rows_entry(
            geometry,
            device,
            row_cache,
            tokens.len(),
            range_count,
            *context_capacity,
        )?;
        let position =
            i32::try_from(state.position()).map_err(|_| "rotary position exceeds index domain")?;
        let coordinates = match conditioned {
            Some(input) => input.coordinates.to_vec(),
            None => tokens
                .iter()
                .enumerate()
                .map(|(row, _)| [position + row as i32; 4])
                .collect::<Vec<_>>(),
        };
        rows.coordinates
            .write_from_host(&tensor_bytes(coordinates.into_iter().flatten()))?;
        let visible = if history_ranges.is_empty() {
            vec![(0, 0)]
        } else {
            history_ranges
        };
        rows.visible
            .write_from_host(&tensor_bytes(tokens.iter().flat_map(|_| {
                visible
                    .iter()
                    .flat_map(|&(start, count)| [start as i32, (start + count) as i32])
            })))?;
        rows.tokens
            .write_from_host(&tensor_bytes(tokens.iter().map(|&token| token as i32)))?;

        let mut observations = collect_observations.then(Vec::new);
        let mut advance = state.begin(tokens.len())?;
        let mut state_results = Vec::new();
        let mut resolved_state_results = Vec::new();
        let mut pending_readout = None;
        advance.execute(|transition| {
            rows.destinations.write_from_host(&tensor_bytes(
                transition.destinations.iter().map(|&value| value as i32),
            ))?;
            let mut workflow = device.workflow();
            let mut hidden = workflow
                .enqueue(
                    &embedding.kernel,
                    kernels::qwen_embedding_rows::WorkflowArgs {
                        table: embedding.table.tensor().into(),
                        tokens: (&rows.tokens).into(),
                    },
                )?
                .value;
            if let Some(input) = conditioned {
                for overlay in input.overlays {
                    conditioning.enqueue(&mut workflow, overlay, &hidden)?;
                }
            }
            for block in blocks {
                let mixed = match &block.mixer {
                    Mixer::Attention(mixer) => {
                        let index = block.state_index;
                        let mut history_key = transition.history[index].clone();
                        let mut history_value = transition.history[index + 1].clone();
                        workflow
                            .enqueue(
                                &mixer.kernel,
                                kernels::qwen_attention_sequence::WorkflowArgs {
                                    hidden: (&hidden).into(),
                                    input_norm: mixer.input_norm.tensor().into(),
                                    query_gate_weight: mixer.query_gate.tensor().into(),
                                    key_weight: mixer.key.tensor().into(),
                                    value_weight: mixer.value.tensor().into(),
                                    query_norm: mixer.query_norm.tensor().into(),
                                    key_norm: mixer.key_norm.tensor().into(),
                                    output_weight: mixer.output.tensor().into(),
                                    coordinates: (&rows.coordinates).into(),
                                    rotary_components: (&mixer.rotary_components).into(),
                                    visible: (&rows.visible).into(),
                                    history_key: (&mut history_key).into(),
                                    history_value: (&mut history_value).into(),
                                    destinations: (&rows.destinations).into(),
                                    base: mixer.base,
                                    epsilon: mixer.epsilon,
                                    scale: mixer.scale,
                                },
                            )?
                            .value
                    }
                    Mixer::Recurrent(mixer) => {
                        let index = block.state_index;
                        let result = workflow.enqueue(
                            &mixer.kernel,
                            kernels::qwen_recurrent_sequence::WorkflowArgs {
                                hidden: (&hidden).into(),
                                input_norm: mixer.input_norm.tensor().into(),
                                qkv_weight: mixer.qkv.tensor().into(),
                                gate_weight: mixer.gate.tensor().into(),
                                alpha_weight: mixer.alpha.tensor().into(),
                                beta_weight: mixer.beta.tensor().into(),
                                convolution: mixer.convolution.tensor().into(),
                                rate: mixer.rate.tensor().into(),
                                time_bias: mixer.time_bias.tensor().into(),
                                recurrent_norm: mixer.norm.tensor().into(),
                                output_weight: mixer.output.tensor().into(),
                                window: (&transition.previous[index]).into(),
                                delta: (&transition.previous[index + 1]).into(),
                                epsilon: mixer.epsilon,
                                preparation_epsilon: mixer.preparation_epsilon,
                                grouped: mixer.grouped,
                            },
                        )?;
                        let mixed = result.r2.clone();
                        state_results.push((index, result));
                        mixed
                    }
                };
                hidden = match &block.feedforward {
                    FeedForward::Dense(feedforward) => {
                        workflow
                            .enqueue(
                                &feedforward.kernel,
                                kernels::qwen_dense_suffix::WorkflowArgs {
                                    residual: (&mixed).into(),
                                    norm: feedforward.norm.tensor().into(),
                                    gate_weight: feedforward.gate.tensor().into(),
                                    up_weight: feedforward.up.tensor().into(),
                                    down_weight: feedforward.down.tensor().into(),
                                    eps: feedforward.epsilon,
                                },
                            )?
                            .r6
                    }
                    FeedForward::Routed(feedforward) => {
                        let row_count = u64::try_from(tokens.len())
                            .map_err(|_| "routed row count exceeds u64")?;
                        let mut routes = Tensor::zeros(
                            device,
                            Element::i32(),
                            &[row_count, feedforward.selected],
                        )?;
                        let mut scores = Tensor::zeros(
                            device,
                            Element::f32(),
                            &[row_count, feedforward.selected],
                        )?;
                        workflow
                            .enqueue(
                                &feedforward.kernel,
                                kernels::qwen_routed_suffix::WorkflowArgs {
                                    residual: (&mixed).into(),
                                    norm: feedforward.norm.tensor().into(),
                                    router: feedforward.router.tensor().into(),
                                    shared_router: feedforward.shared_router.tensor().into(),
                                    expert_gate: feedforward.expert_gate.tensor().into(),
                                    expert_up: feedforward.expert_up.tensor().into(),
                                    expert_down: feedforward.expert_down.tensor().into(),
                                    shared_gate: feedforward.shared_gate.tensor().into(),
                                    shared_up: feedforward.shared_up.tensor().into(),
                                    shared_down: feedforward.shared_down.tensor().into(),
                                    routes: (&mut routes).into(),
                                    scores: (&mut scores).into(),
                                    eps: feedforward.epsilon,
                                    normalize: i32::from(feedforward.normalize),
                                },
                            )?
                            .r14
                    }
                };
            }
            if let Some(selected) = selected.as_mut() {
                pending_readout = Some(PendingReadout::Selected(workflow.enqueue(
                    &readout_kernels.selected,
                    kernels::qwen_readout_selected::WorkflowArgs {
                        hidden: (&hidden).into(),
                        norm: readout_kernels.norm.tensor().into(),
                        weight: readout_kernels.weight.tensor().into(),
                        selected: (&selected.ids).into(),
                        epsilon: readout_kernels.epsilon,
                    },
                )?));
            } else if !matches!(&readout, Readout::StateOnly | Readout::Selected(_)) {
                pending_readout = Some(PendingReadout::Rows(workflow.enqueue(
                    &readout_kernels.rows,
                    kernels::qwen_readout_rows::WorkflowArgs {
                        hidden: (&hidden).into(),
                        norm: readout_kernels.norm.tensor().into(),
                        weight: readout_kernels.weight.tensor().into(),
                        epsilon: readout_kernels.epsilon,
                    },
                )?));
            }
            let started = Instant::now();
            let completion = workflow.submit()?;
            if let Some(observations) = &mut observations {
                observations.push(DecoderStepObservation {
                    stage: "workflow".into(),
                    block: None,
                    entry: "qwen_decoder_step".into(),
                    execution: StageExecutionObservation {
                        host_seconds: started.elapsed().as_secs_f64(),
                    },
                });
            }
            for (index, result) in state_results.drain(..) {
                let result =
                    completion.resolve::<kernels::qwen_recurrent_sequence::Entry>(result)?;
                resolved_state_results.push((index, result.r0));
                resolved_state_results.push((index + 1, result.r1));
            }
            match pending_readout.take() {
                Some(PendingReadout::Rows(result)) => {
                    rows.logits = completion
                        .resolve::<kernels::qwen_readout_rows::Entry>(result)?
                        .value;
                }
                Some(PendingReadout::Selected(result)) => {
                    if let Some(selected) = selected.as_mut() {
                        selected.logits = completion
                            .resolve::<kernels::qwen_readout_selected::Entry>(result)?
                            .value;
                    }
                }
                None => {}
            }
            Ok(())
        })?;
        for (index, tensor) in resolved_state_results {
            advance.replace_following(index, tensor)?;
        }
        let output = match readout {
            Readout::StateOnly => ReadoutOutput::StateOnly,
            Readout::Selected(_) => ReadoutOutput::Selected(
                selected
                    .map(|selection| selection.logits.read_to_host())
                    .transpose()?
                    .unwrap_or_default()
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect(),
            ),
            Readout::Logits => ReadoutOutput::Logits(
                rows.logits
                    .read_to_host()?
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect(),
            ),
            Readout::Sample {
                mask,
                sampling,
                seed,
                position,
            } => ReadoutOutput::Sample(sampler.sample(
                &rows.logits,
                mask,
                sampling,
                seed,
                position,
            )?),
        };
        Ok((
            ExecutedAdvance { advance, output },
            observations.unwrap_or_default(),
        ))
    }
}
