//! Single-row dense and routed Qwen numerical decoder. Dense KV history and synchronous
//! execution are explicit baseline limitations, not a complete serving engine.
use super::{Description, FeedForwardWeights, Geometry, HeadMapping, MixerWeights};
use crate::{
    execution::{Composition, CompositionSpec},
    state::{ComponentSpec, SequenceState, StateAdvance, StateStore},
    weights::{descriptor::WeightDescriptor, residency::ResidentWeight},
};
use seismic_lang::types::{DType, Elem};
use seismic_runtime::{
    Buffer, Device, ExecutionObservation,
    plan::{Diagnostic, PlanCompiler, Settings, StepObservation, Submission},
};
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};
fn names(xs: &[&str]) -> HashSet<String> {
    xs.iter().map(|s| (*s).into()).collect()
}
fn shape(xs: &[(&str, u64)]) -> Result<HashMap<String, i64>, String> {
    xs.iter()
        .map(|(s, n)| {
            Ok((
                (*s).into(),
                i64::try_from(*n).map_err(|_| "dimension exceeds index range")?,
            ))
        })
        .collect()
}
fn weights(xs: Vec<(&str, ResidentWeight)>) -> HashMap<String, ResidentWeight> {
    xs.into_iter().map(|(n, w)| (n.into(), w)).collect()
}
fn scalar(xs: &[(&str, f64)]) -> HashMap<String, f64> {
    xs.iter().map(|(n, v)| ((*n).into(), *v)).collect()
}
struct Block {
    mixer: Composition,
    feedforward: Composition,
    state_index: usize,
    attention: bool,
}
pub struct Decoder {
    geometry: Geometry,
    embedding: Composition,
    blocks: Vec<Block>,
    readout: Composition,
    store: Rc<StateStore>,
    hidden: Buffer,
    logits: Buffer,
    coordinates: Buffer,
    visible: Buffer,
    compiled_kernels: usize,
}
#[derive(Clone, Debug)]
pub struct DecoderStepObservation {
    pub stage: String,
    pub block: Option<usize>,
    pub step: StepObservation,
}
fn execute_stage(
    composition: &mut Composition,
    tensors: &HashMap<String, Buffer>,
    scalars: &HashMap<String, f64>,
    stage: &str,
    block: Option<usize>,
    observations: &mut Option<Vec<DecoderStepObservation>>,
    submission: &mut Option<Submission>,
) -> Result<(), String> {
    if let Some(submission) = submission {
        submission.append(composition.prepare(tensors, scalars)?);
        Ok(())
    } else if let Some(observations) = observations {
        observations.extend(
            composition
                .execute_observed(tensors, scalars)?
                .into_iter()
                .map(|step| DecoderStepObservation {
                    stage: stage.into(),
                    block,
                    step,
                }),
        );
        Ok(())
    } else {
        composition.execute(tensors, scalars)
    }
}
pub struct DecodedAdvance<'a> {
    advance: StateAdvance<'a>,
    logits: Vec<f32>,
}
impl DecodedAdvance<'_> {
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn commit(self) -> Result<(), String> {
        self.advance.commit()
    }
    pub fn abort(self) {
        self.advance.abort()
    }
}
impl Decoder {
    /// Import callbacks resolve container storage and preserve each declared
    /// target publication. Packed resident representations remain packed.
    pub fn compile(
        device: Rc<Device>,
        description: &Description,
        import: impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, String>,
        settings: Settings,
        context_capacity: usize,
        max_sequences: usize,
    ) -> Result<Self, String> {
        Self::compile_with(
            device,
            description,
            import,
            Ok(settings),
            context_capacity,
            max_sequences,
        )
    }
    /// Explicit fixed assignments for reference and native qualification only.
    pub fn compile_diagnostic(
        device: Rc<Device>,
        description: &Description,
        import: impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, String>,
        choices: Diagnostic,
        context_capacity: usize,
        max_sequences: usize,
    ) -> Result<Self, String> {
        Self::compile_with(
            device,
            description,
            import,
            Err(choices),
            context_capacity,
            max_sequences,
        )
    }
    fn compile_with(
        device: Rc<Device>,
        description: &Description,
        mut import: impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, String>,
        settings: Result<Settings, Diagnostic>,
        context_capacity: usize,
        max_sequences: usize,
    ) -> Result<Self, String> {
        let g = &description.geometry;
        g.validate().map_err(|e| e.to_string())?;
        if description.blocks.len() != g.layers.len() {
            return Err("decoder requires complete block descriptors".into());
        }
        if context_capacity == 0
            || context_capacity as u64 > g.context_limit
            || context_capacity > i32::MAX as usize
            || max_sequences == 0
        {
            return Err("invalid decoder context or sequence capacity".into());
        }
        let history_capacity = context_capacity
            .checked_mul(max_sequences)
            .filter(|n| *n <= i32::MAX as usize)
            .ok_or("history capacity overflow")?;
        let program = super::program::program()?;
        let mut compiler = match settings {
            Ok(settings) => PlanCompiler::new(&device, &program, settings),
            Err(choices) => {
                PlanCompiler::diagnostic(&device, &program, choices.lowering, choices.candidate)
            }
        };
        let activation = g.activation_dtype;
        let mut bound =
            |entry: &str, shapes, weights, external: &[&str], intermediates: &[&str], scalars| {
                Composition::compile(
                    &mut compiler,
                    CompositionSpec {
                        entry: entry.into(),
                        shapes,
                        elements: HashMap::from([("A".into(), Elem::Dtype(activation))]),
                        weights,
                        external: names(external),
                        intermediates: names(intermediates),
                        scalars,
                    },
                )
            };
        let embedding = bound(
            "qwen_embedding",
            shape(&[("V", g.vocabulary), ("D", g.hidden)])?,
            weights(vec![("table", import(&description.embedding, activation)?)]),
            &["out"],
            &["embedded"],
            HashMap::new(),
        )?;
        let mut blocks = Vec::new();
        let mut components = Vec::new();
        let mut history_rows = Vec::new();
        for (index, block) in description.blocks.iter().enumerate() {
            let input_norm = import(&block.input_norm, activation)?;
            let (mixer, state_index, attention) = match &block.mixer {
                MixerWeights::Attention(a) => {
                    let state_index = history_rows.len();
                    let row_bytes = usize::try_from(
                        g.kv_heads
                            .checked_mul(g.attention_width)
                            .and_then(|n| n.checked_mul(u64::from(activation.bytes())))
                            .ok_or("KV geometry overflow")?,
                    )
                    .map_err(|_| "KV row exceeds address range")?;
                    history_rows.extend([row_bytes, row_bytes]);
                    let mixer = bound(
                        "qwen_attention_step",
                        shape(&[
                            ("D", g.hidden),
                            ("T", history_capacity as u64),
                            ("H", g.attention_heads),
                            ("KV", g.kv_heads),
                            ("P", g.rotary_width / 2),
                            ("S", g.attention_width - g.rotary_width),
                            ("SH", g.rotary_sections[1]),
                            ("SW", g.rotary_sections[2]),
                        ])?,
                        weights(vec![
                            ("input_norm", input_norm),
                            ("query_gate_weight", import(&a.query_gate, activation)?),
                            ("key_weight", import(&a.key, activation)?),
                            ("value_weight", import(&a.value, activation)?),
                            ("query_norm", import(&a.query_norm, DType::F32)?),
                            ("key_norm", import(&a.key_norm, DType::F32)?),
                            ("output_weight", import(&a.output, activation)?),
                        ]),
                        &[
                            "hidden",
                            "out",
                            "coordinates",
                            "visible",
                            "history_key",
                            "history_value",
                        ],
                        &[
                            "normalized",
                            "query_gate",
                            "key",
                            "value",
                            "query",
                            "prepared_key",
                            "gate",
                            "attended",
                            "activated",
                            "gated",
                            "projected",
                        ],
                        scalar(&[
                            ("base", g.rotary_base),
                            ("epsilon", g.epsilon),
                            ("scale", 1.0 / (g.attention_width as f64).sqrt()),
                        ]),
                    )?;
                    (mixer, state_index, true)
                }
                MixerWeights::Recurrent(r) => {
                    let state_index = components.len();
                    let to_usize =
                        |n| usize::try_from(n).map_err(|_| "component exceeds address range");
                    components.push(ComponentSpec {
                        shape: vec![
                            to_usize(g.convolution_width - 1)?,
                            to_usize(g.recurrent_channels().map_err(|e| e.to_string())?)?,
                        ],
                        dtype: activation,
                    });
                    components.push(ComponentSpec {
                        shape: vec![
                            to_usize(g.recurrent_value_heads)?,
                            to_usize(g.recurrent_width)?,
                            to_usize(g.recurrent_width)?,
                        ],
                        dtype: DType::F32,
                    });
                    let mixer = bound(
                        "qwen_recurrent_step",
                        shape(&[
                            ("H", g.hidden),
                            ("NK", g.recurrent_key_heads),
                            ("GV", g.recurrent_value_heads / g.recurrent_key_heads),
                            ("W", g.recurrent_width),
                            ("C", g.convolution_width),
                        ])?,
                        weights(vec![
                            ("input_norm", input_norm),
                            ("qkv_weight", import(&r.query_key_value, activation)?),
                            ("gate_weight", import(&r.gate, activation)?),
                            ("alpha_weight", import(&r.alpha, activation)?),
                            ("beta_weight", import(&r.beta, activation)?),
                            ("convolution", import(&r.convolution, DType::F32)?),
                            ("rate", import(&r.decay, DType::F32)?),
                            ("time_bias", import(&r.time_bias, DType::F32)?),
                            ("recurrent_norm", import(&r.norm, activation)?),
                            ("output_weight", import(&r.output, activation)?),
                        ]),
                        &[
                            "hidden",
                            "out",
                            "window",
                            "delta",
                            "next_window",
                            "next_delta",
                        ],
                        &[
                            "normalized",
                            "projected",
                            "gate",
                            "alpha",
                            "beta_input",
                            "qkv",
                            "beta",
                            "decay",
                            "mixed",
                            "mixed_norm",
                            "activated",
                            "gated",
                            "output_projection",
                        ],
                        scalar(&[
                            ("epsilon", g.epsilon),
                            ("preparation_epsilon", g.epsilon * g.recurrent_width as f64),
                            (
                                "grouped",
                                f64::from(g.recurrent_head_mapping == HeadMapping::Grouped),
                            ),
                        ]),
                    )?;
                    (mixer, state_index, false)
                }
            };
            if (g.layers[index] == super::MixerKind::Attention) != attention {
                return Err("block descriptor disagrees with layer order".into());
            }
            let feedforward = match &block.feedforward {
                FeedForwardWeights::Dense(ff) => {
                    if g.experts.is_some() {
                        return Err("dense feedforward disagrees with routed geometry".into());
                    }
                    bound(
                        "qwen_dense_suffix",
                        shape(&[("M", 1), ("H", g.hidden), ("F", g.intermediate)])?,
                        weights(vec![
                            ("norm", import(&block.feedforward_norm, activation)?),
                            ("gate_weight", import(&ff.gate, activation)?),
                            ("up_weight", import(&ff.up, activation)?),
                            ("down_weight", import(&ff.down, activation)?),
                        ]),
                        &["residual", "out"],
                        &[
                            "normalized",
                            "gate",
                            "up",
                            "activated",
                            "product",
                            "projected",
                        ],
                        scalar(&[("eps", g.epsilon)]),
                    )?
                }
                FeedForwardWeights::Routed(ff) => {
                    let experts = g
                        .experts
                        .as_ref()
                        .ok_or("routed feedforward lacks expert geometry")?;
                    bound(
                        "qwen_routed_suffix",
                        shape(&[
                            ("M", 1),
                            ("H", g.hidden),
                            ("E", experts.count),
                            ("K", experts.selected),
                            ("F", experts.intermediate),
                            ("S", experts.shared_intermediate),
                        ])?,
                        weights(vec![
                            ("norm", import(&block.feedforward_norm, activation)?),
                            ("router", import(&ff.router, activation)?),
                            ("shared_router", import(&ff.shared_router, DType::F32)?),
                            ("expert_gate", import(&ff.expert_gate, activation)?),
                            ("expert_up", import(&ff.expert_up, activation)?),
                            ("expert_down", import(&ff.expert_down, activation)?),
                            ("shared_gate", import(&ff.shared_gate, activation)?),
                            ("shared_up", import(&ff.shared_up, activation)?),
                            ("shared_down", import(&ff.shared_down, activation)?),
                        ]),
                        &["residual", "out"],
                        &[
                            "normalized",
                            "logits",
                            "routes",
                            "scores",
                            "gate",
                            "up",
                            "product",
                            "projected",
                            "sg",
                            "su",
                            "sa",
                            "sp",
                            "shared",
                            "coefficient",
                        ],
                        scalar(&[
                            ("eps", g.epsilon),
                            ("normalize", f64::from(experts.normalize_selected)),
                        ]),
                    )?
                }
            };
            blocks.push(Block {
                mixer,
                feedforward,
                state_index,
                attention,
            });
        }
        let readout = bound(
            "qwen_readout",
            shape(&[("V", g.vocabulary), ("D", g.hidden)])?,
            weights(vec![
                ("norm", import(&description.output_norm, activation)?),
                ("weight", import(&description.output, activation)?),
            ]),
            &["hidden", "logits"],
            &["normalized"],
            scalar(&[("epsilon", g.epsilon)]),
        )?;
        let compiled_kernels = compiler.kernel_count();
        let hidden_bytes = usize::try_from(g.hidden)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or("residual allocation overflow")?;
        let logits_bytes = usize::try_from(g.vocabulary)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or("logits allocation overflow")?;
        let hidden = device.buffer(hidden_bytes)?;
        let logits = device.buffer(logits_bytes)?;
        let coordinates = device.buffer(16)?;
        let visible = device.buffer(8)?;
        let store = StateStore::new(
            device,
            context_capacity,
            history_capacity,
            history_rows,
            components,
        )?;
        Ok(Self {
            geometry: g.clone(),
            embedding,
            blocks,
            readout,
            store,
            hidden,
            logits,
            coordinates,
            visible,
            compiled_kernels,
        })
    }
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    pub fn state_store(&self) -> &Rc<StateStore> {
        &self.store
    }
    pub fn compiled_kernel_count(&self) -> usize {
        self.compiled_kernels
    }
    pub fn propose<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<DecodedAdvance<'a>, String> {
        self.propose_impl(state, token, false, false)
            .map(|(advance, _, _)| advance)
    }
    pub fn propose_observed<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<(DecodedAdvance<'a>, Vec<DecoderStepObservation>), String> {
        self.propose_impl(state, token, true, false)
            .map(|(advance, steps, _)| (advance, steps))
    }
    pub fn propose_batched<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<(DecodedAdvance<'a>, ExecutionObservation), String> {
        self.propose_impl(state, token, false, true)
            .map(|(advance, _, batch)| {
                (
                    advance,
                    batch.expect("batched execution records completion"),
                )
            })
    }
    fn propose_impl<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
        observed: bool,
        batched: bool,
    ) -> Result<
        (
            DecodedAdvance<'a>,
            Vec<DecoderStepObservation>,
            Option<ExecutionObservation>,
        ),
        String,
    > {
        let mut observations = observed.then(Vec::new);
        let mut submission = batched.then(Submission::default);
        let mut batch_observation = None;
        if !state.belongs_to(&self.store) {
            return Err("sequence belongs to another decoder state store".into());
        }
        if u64::from(token) >= self.geometry.vocabulary {
            return Err("token is outside vocabulary".into());
        }
        let ranges = state.history_ranges();
        if ranges.len() > 1 {
            return Err(
                "dense decode baseline currently requires one visible history range".into(),
            );
        }
        let position =
            i32::try_from(state.position()).map_err(|_| "rotary position exceeds index domain")?;
        self.coordinates.write(
            &[position; 4]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let (start, count) = ranges.first().copied().unwrap_or((0, 0));
        self.visible.write(
            &[start as i32, (start + count) as i32]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let mut advance = state.begin(1)?;
        advance.execute(|transition| {
            execute_stage(
                &mut self.embedding,
                &HashMap::from([("out".into(), self.hidden.clone())]),
                &scalar(&[("token", f64::from(token))]),
                "embedding",
                None,
                &mut observations,
                &mut submission,
            )?;
            for (block_index, block) in self.blocks.iter_mut().enumerate() {
                let mut tensors = HashMap::from([
                    ("hidden".into(), self.hidden.clone()),
                    ("out".into(), self.hidden.clone()),
                ]);
                let parameters = if block.attention {
                    let i = block.state_index;
                    tensors.extend([
                        ("coordinates".into(), self.coordinates.clone()),
                        ("visible".into(), self.visible.clone()),
                        ("history_key".into(), transition.history[i].clone()),
                        ("history_value".into(), transition.history[i + 1].clone()),
                    ]);
                    scalar(&[("destination", transition.destinations[0] as f64)])
                } else {
                    let i = block.state_index;
                    tensors.extend([
                        ("window".into(), transition.previous[i].clone()),
                        ("delta".into(), transition.previous[i + 1].clone()),
                        ("next_window".into(), transition.following[i].clone()),
                        ("next_delta".into(), transition.following[i + 1].clone()),
                    ]);
                    HashMap::new()
                };
                execute_stage(
                    &mut block.mixer,
                    &tensors,
                    &parameters,
                    "mixer",
                    Some(block_index),
                    &mut observations,
                    &mut submission,
                )?;
                execute_stage(
                    &mut block.feedforward,
                    &HashMap::from([
                        ("residual".into(), self.hidden.clone()),
                        ("out".into(), self.hidden.clone()),
                    ]),
                    &HashMap::new(),
                    "feedforward",
                    Some(block_index),
                    &mut observations,
                    &mut submission,
                )?;
            }
            execute_stage(
                &mut self.readout,
                &HashMap::from([
                    ("hidden".into(), self.hidden.clone()),
                    ("logits".into(), self.logits.clone()),
                ]),
                &HashMap::new(),
                "readout",
                None,
                &mut observations,
                &mut submission,
            )?;
            if let Some(submission) = &mut submission {
                batch_observation = Some(submission.execute_batched()?);
            }
            Ok(())
        })?;
        let mut bytes = vec![0; self.logits.len()];
        self.logits.read(&mut bytes)?;
        let logits = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok((
            DecodedAdvance { advance, logits },
            observations.unwrap_or_default(),
            batch_observation,
        ))
    }
}
