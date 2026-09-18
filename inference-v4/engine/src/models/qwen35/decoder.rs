//! Qwen sequence forwards and multi-request generation. Sequence successors
//! remain private until shared completion and independent request acceptance.
mod packed;
use super::{Description, FeedForwardWeights, Geometry, HeadMapping, MixerWeights};
use crate::{
    execution::{Composition, CompositionSpec},
    generation::{
        sampling::{Sampler, Selection},
        Proposal, Sampling,
    },
    models::sequence::{Advance, OwnedSequence, SequenceWork},
    state::{ComponentSpec, SequenceState, StateAdvance, StateStore},
    weights::{descriptor::WeightDescriptor, residency::ResidentWeight},
};
use packed::PackedRows;
use seismic_lang::types::{DType, Elem};
use seismic_runtime::{
    plan::{PlanCompiler, Settings, StepObservation, Submission},
    Buffer, Device, Error, ExecutionObservation,
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
struct Rows {
    embedding: Composition,
    blocks: Vec<Block>,
    readout: Composition,
    hidden: Buffer,
    logits: Buffer,
    coordinates: Buffer,
    visible: Buffer,
    tokens: Buffer,
    destinations: Buffer,
}
struct SelectedRows {
    readout: Composition,
    ids: Buffer,
    logits: Buffer,
}
pub struct Decoder {
    geometry: Geometry,
    store: Rc<StateStore>,
    device: Rc<Device>,
    program: seismic_lang::program::Program,
    settings: Settings,
    context_capacity: usize,
    rows: HashMap<(usize, usize), Rows>,
    sampler: Option<Sampler>,
    selected_template: Composition,
    selected: HashMap<(usize, usize), SelectedRows>,
    packed: HashMap<usize, PackedRows>,
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
/// Explicit numerical output; state-only execution omits the vocabulary projection.
pub enum Readout<'a> {
    StateOnly,
    Logits,
    /// Final-row logits in exactly this order. Duplicates are retained; an empty
    /// selection advances state without projecting any vocabulary rows.
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
        ReadoutOutput::Sample(Selection::Nonfinite) => Err("nonfinite sampling distribution".into()),
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
        self.advance.abort()
    }
}
pub struct ConditionedAdvance<'a> {
    advance: ExecutedAdvance<'a>,
    input: &'a mut super::inputs::InputState,
    next: super::inputs::InputState,
}
impl ConditionedAdvance<'_> {
    pub fn output(&self) -> &ReadoutOutput { self.advance.output() }
    pub fn commit(self) -> Result<(), String> {
        self.advance.commit()?;
        *self.input = self.next;
        Ok(())
    }
    pub fn abort(self) { self.advance.abort(); }
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
        self.advance.abort()
    }
}
/// Logical members of one numerical preparation; masks are request-local.
pub struct GenerationWork<'a, S = ()> {
    pub sequence: &'a OwnedSequence<S>,
    pub proposal: &'a Proposal,
    pub mask: Option<&'a [u32]>,
}
impl Decoder {
    /// Reserve every sequence successor before executing any member. Selection
    /// outcomes remain independent after all synchronous execution completes.
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
            if work.len() > 1 {
                return self.execute_generation_states(states, work);
            }
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
    /// Conditioned members retain semantic successors with the numerical fork.
    /// Each member executes ordinary Seismic compositions; acceptance remains
    /// independent after every member has completed. Packed conditioned rows
    /// are not yet implemented by this entry.
    pub fn prepare_conditioned_generation_batch(
        &mut self,
        work: &[GenerationWork<'_, super::inputs::InputState>],
    ) -> Result<Vec<Box<dyn Advance>>, Error> {
        let sequences = work.iter().map(|row| SequenceWork {
            sequence: row.sequence, position: row.proposal.position(), count: row.proposal.tokens().len(),
        }).collect::<Vec<_>>();
        OwnedSequence::prepare_completed_batch_with_semantics(&sequences, |states| {
            let mut selected = Vec::with_capacity(work.len());
            for (row, (state, input)) in work.iter().zip(states.iter_mut()) {
                let tokens: Vec<_> = row.proposal.tokens().iter().map(|token| token.0).collect();
                let readout = if row.proposal.needs_sample() {
                    Readout::Sample { mask: row.mask, sampling: row.proposal.sampling(), seed: row.proposal.seed(), position: row.proposal.sample_position() }
                } else { Readout::StateOnly };
                let advance = self.execute_conditioned(state, input, &tokens, readout)?;
                let outcome = generation_selection(advance.output());
                advance.commit()?;
                selected.push(outcome);
            }
            Ok(selected)
        })
    }
    /// Import callbacks resolve container storage and preserve each declared
    /// target publication. Packed resident representations remain packed.
    pub fn compile(
        device: Rc<Device>,
        description: &Description,
        mut import: impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, String>,
        settings: Settings,
        context_capacity: usize,
        max_sequences: usize,
    ) -> Result<Self, Error> {
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
        let mut compiler = PlanCompiler::new(&device, &program, settings.clone());
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
            "qwen_embedding_rows",
            shape(&[("M", 1), ("V", g.vocabulary), ("D", g.hidden)])?,
            weights(vec![("table", import(&description.embedding, activation)?)]),
            &["tokens", "out"],
            &["embedded"],
            HashMap::new(),
        )?
        .control_domain("tokens", seismic_runtime::tuner::IntegerRange { min: 0, max: i128::from(g.vocabulary) - 1, stride: 1 })?;
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
                        "qwen_attention_sequence",
                        shape(&[
                            ("M", 1),
                            ("D", g.hidden),
                            ("T", history_capacity as u64),
                            ("R", 1),
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
                            "destinations",
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
                    let mixer =
                        mixer.control_inputs(&["visible"])?
                            .control_domain("coordinates", seismic_runtime::tuner::IntegerRange { min: 0, max: i128::from(i32::MAX), stride: 1 })?
                            .control_domain("destinations", seismic_runtime::tuner::IntegerRange { min: 0, max: history_capacity as i128 - 1, stride: 1 })?;
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
                        "qwen_recurrent_sequence",
                        shape(&[
                            ("M", 1),
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
        let readout_weights = weights(vec![
            ("norm", import(&description.output_norm, activation)?),
            ("weight", import(&description.output, activation)?),
        ]);
        let selected_template = bound(
            "qwen_readout_selected",
            shape(&[("M", 1), ("V", g.vocabulary), ("D", g.hidden), ("S", 1)])?,
            readout_weights.clone(),
            &["hidden", "selected", "logits"],
            &["last", "normalized"],
            scalar(&[("epsilon", g.epsilon)]),
        )?
        .control_domain("selected", seismic_runtime::tuner::IntegerRange { min: 0, max: i128::from(g.vocabulary) - 1, stride: 1 })?;
        let readout = bound(
            "qwen_readout_rows",
            shape(&[("M", 1), ("V", g.vocabulary), ("D", g.hidden)])?,
            readout_weights,
            &["hidden", "logits"],
            &["last", "normalized"],
            scalar(&[("epsilon", g.epsilon)]),
        )?;
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
            device.clone(),
            context_capacity,
            history_capacity,
            history_rows,
            components,
        )?;
        let tokens = device.buffer(4)?;
        let destinations = device.buffer(4)?;
        let rows = Rows {
            embedding,
            blocks,
            readout,
            hidden,
            logits,
            coordinates,
            visible,
            tokens,
            destinations,
        };
        Ok(Self {
            geometry: g.clone(),
            store,
            device,
            program,
            settings,
            context_capacity,
            rows: HashMap::from([((1, 1), rows)]),
            sampler: None,
            selected_template,
            selected: HashMap::new(),
            packed: HashMap::new(),
        })
    }
    fn ensure_selected(&mut self, rows: usize, ids: &[u32]) -> Result<(), Error> {
        if ids
            .iter()
            .any(|&id| u64::from(id) >= self.geometry.vocabulary || id > i32::MAX as u32)
        {
            return Err("selected readout token is outside vocabulary".into());
        }
        if ids.is_empty() {
            return Ok(());
        }
        let key = (rows, ids.len());
        if !self.selected.contains_key(&key) {
            let mut compiler =
                PlanCompiler::new(&self.device, &self.program, self.settings.clone());
            let readout = self
                .selected_template
                .with_dimensions(&mut compiler, &[("M", rows), ("S", ids.len())])?;
            let bytes = ids
                .len()
                .checked_mul(4)
                .ok_or("selected readout size overflow")?;
            self.selected.insert(
                key,
                SelectedRows {
                    readout,
                    ids: self.device.buffer(bytes)?,
                    logits: self.device.buffer(bytes)?,
                },
            );
        }
        self.selected
            .get(&key)
            .expect("selection prepared")
            .ids
            .write(
                &ids.iter()
                    .flat_map(|&id| (id as i32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        Ok(())
    }
    fn ensure_rows(&mut self, count: usize, ranges: usize) -> Result<(), Error> {
        if self.rows.contains_key(&(count, ranges)) {
            return Ok(());
        }
        if count == 0 || ranges == 0 || count as u64 > self.geometry.context_limit {
            return Err("invalid forward row count".into());
        }
        let template = self.rows.get(&(1, 1)).expect("decode geometry exists");
        let mut compiler = PlanCompiler::new(&self.device, &self.program, self.settings.clone());
        let blocks = template
            .blocks
            .iter()
            .map(|b| {
                let dimensions = [("M", count), ("R", ranges)];
                Ok(Block {
                    mixer: b.mixer.with_dimensions(
                        &mut compiler,
                        &dimensions[..if b.attention { 2 } else { 1 }],
                    )?,
                    feedforward: b
                        .feedforward
                        .with_dimensions(&mut compiler, &[("M", count)])?,
                    state_index: b.state_index,
                    attention: b.attention,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let bytes = |width: usize| count.checked_mul(width).ok_or("forward buffer overflow");
        let rows = Rows {
            embedding: template
                .embedding
                .with_dimensions(&mut compiler, &[("M", count)])?,
            readout: template
                .readout
                .with_dimensions(&mut compiler, &[("M", count)])?,
            blocks,
            hidden: self.device.buffer(bytes(template.hidden.len())?)?,
            logits: self.device.buffer(template.logits.len())?,
            coordinates: self.device.buffer(bytes(16)?)?,
            visible: self.device.buffer(bytes(
                ranges.checked_mul(8).ok_or("visibility buffer overflow")?,
            )?)?,
            tokens: self.device.buffer(bytes(4)?)?,
            destinations: self.device.buffer(bytes(4)?)?,
        };
        self.rows.insert((count, ranges), rows);
        Ok(())
    }
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    pub fn context_capacity(&self) -> usize {
        self.context_capacity
    }
    pub fn memory_usage(&self) -> seismic_runtime::memory::Usage {
        self.device.memory_usage()
    }
    /// Drop idle row specializations and sampling storage. The single-row
    /// composition remains the source for later row specialization.
    pub fn reclaim_idle(&mut self) -> Result<usize, String> {
        let before = self.device.memory_usage().charged;
        self.rows.retain(|geometry, _| *geometry == (1, 1));
        self.sampler.take();
        self.selected.clear();
        self.packed.clear();
        self.store.release_idle()?;
        Ok(before - self.device.memory_usage().charged)
    }
    pub fn state_store(&self) -> &Rc<StateStore> {
        &self.store
    }
    pub fn compiled_kernel_count(&self) -> usize {
        let mut unique: Vec<&Composition> = Vec::new();
        for rows in self.rows.values() {
            for composition in std::iter::once(&rows.embedding)
                .chain(rows.blocks.iter().flat_map(|b| [&b.mixer, &b.feedforward]))
                .chain(std::iter::once(&rows.readout))
            {
                if !unique
                    .iter()
                    .any(|previous| previous.shares_compilation(composition))
                {
                    unique.push(composition);
                }
            }
        }
        for composition in self
            .packed
            .values()
            .flat_map(PackedRows::compositions)
            .chain(std::iter::once(&self.selected_template))
            .chain(self.selected.values().map(|rows| &rows.readout))
        {
            if !unique
                .iter()
                .any(|previous| previous.shares_compilation(composition))
            {
                unique.push(composition);
            }
        }
        unique.iter().map(|c| c.kernel_count()).sum::<usize>()
            + self.sampler.as_ref().map_or(0, Sampler::kernel_count)
    }
    pub fn propose<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<DecodedAdvance<'a>, Error> {
        self.propose_impl(state, &[token], false, false, Readout::Logits, None)
            .map(|(advance, _, _)| advance.decoded())
    }
    pub fn propose_observed<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<(DecodedAdvance<'a>, Vec<DecoderStepObservation>), Error> {
        self.propose_impl(state, &[token], true, false, Readout::Logits, None)
            .map(|(advance, steps, _)| (advance.decoded(), steps))
    }
    pub fn propose_batched<'a>(
        &mut self,
        state: &'a mut SequenceState,
        token: u32,
    ) -> Result<(DecodedAdvance<'a>, ExecutionObservation), Error> {
        self.propose_impl(state, &[token], false, true, Readout::Logits, None)
            .map(|(advance, _, batch)| {
                (
                    advance.decoded(),
                    batch.expect("batched execution records completion"),
                )
            })
    }
    /// Evaluate a complete prompt (or prompt continuation) as one multi-row
    /// forward. Only final-row logits are read back; commit accepts all rows.
    pub fn prefill<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
    ) -> Result<DecodedAdvance<'a>, Error> {
        self.propose_impl(state, tokens, false, false, Readout::Logits, None)
            .map(|(advance, _, _)| advance.decoded())
    }
    pub fn prefill_observed<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
    ) -> Result<(DecodedAdvance<'a>, Vec<DecoderStepObservation>), Error> {
        self.propose_impl(state, tokens, true, false, Readout::Logits, None)
            .map(|(advance, steps, _)| (advance.decoded(), steps))
    }
    pub fn prefill_batched<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
    ) -> Result<(DecodedAdvance<'a>, ExecutionObservation), Error> {
        self.propose_impl(state, tokens, false, true, Readout::Logits, None)
            .map(|(advance, _, batch)| {
                (
                    advance.decoded(),
                    batch.expect("batched execution records completion"),
                )
            })
    }
    /// Complete forward and optional device selection before exposing an advance.
    pub fn execute<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
        readout: Readout<'_>,
    ) -> Result<ExecutedAdvance<'a>, Error> {
        self.propose_impl(state, tokens, false, true, readout, None)
            .map(|(advance, _, _)| advance)
    }
    /// Advance semantic and numerical state as one accepted transaction.
    pub fn execute_conditioned<'a>(
        &mut self, state: &'a mut SequenceState, input: &'a mut super::inputs::InputState,
        tokens: &[u32], readout: Readout<'_>,
    ) -> Result<ConditionedAdvance<'a>, Error> {
        if input.position() != state.position() || input.width() as u64 != self.geometry.hidden || !input.belongs_to(&self.device) {
            return Err("conditioned input differs from decoder continuation or owner".into());
        }
        let assembled = input.assemble(tokens)?;
        let next = input.after(input.position().checked_add(tokens.len()).ok_or("input advance overflow")?)?;
        let (advance, _, _) = self.propose_impl(state, tokens, false, true, readout, Some(&assembled))?;
        Ok(ConditionedAdvance { advance, input, next })
    }
    fn propose_impl<'a>(
        &mut self,
        state: &'a mut SequenceState,
        tokens: &[u32],
        observed: bool,
        batched: bool,
        readout: Readout<'_>,
        conditioned: Option<&super::inputs::Assembled>,
    ) -> Result<
        (
            ExecutedAdvance<'a>,
            Vec<DecoderStepObservation>,
            Option<ExecutionObservation>,
        ),
        Error,
    > {
        let mut observations = observed.then(Vec::new);
        let mut submission = batched.then(Submission::default);
        let mut batch_observation = None;
        if !state.belongs_to(&self.store) {
            return Err("sequence belongs to another decoder state store".into());
        }
        if tokens.is_empty()
            || tokens
                .iter()
                .any(|&t| u64::from(t) >= self.geometry.vocabulary || t > i32::MAX as u32)
        {
            return Err("token is outside vocabulary".into());
        }
        let ranges = state.history_ranges();
        if tokens.len() > self.context_capacity - state.position() {
            return Err("forward exceeds context limit".into());
        }
        if matches!(&readout, Readout::Sample { .. }) && self.sampler.is_none() {
            self.sampler = Some(Sampler::compile(
                &self.device,
                self.geometry.vocabulary as usize,
                self.settings.clone(),
            )?);
        }
        if let Readout::Selected(ids) = &readout {
            self.ensure_selected(tokens.len(), ids)?;
        }
        self.ensure_rows(tokens.len(), ranges.len().max(1))?;
        let mut selected = match &readout {
            Readout::Selected(ids) if !ids.is_empty() => {
                self.selected.get_mut(&(tokens.len(), ids.len()))
            }
            _ => None,
        };
        let rows = self
            .rows
            .get_mut(&(tokens.len(), ranges.len().max(1)))
            .expect("row geometry prepared");
        let position =
            i32::try_from(state.position()).map_err(|_| "rotary position exceeds index domain")?;
        let coordinates: Vec<[i32; 4]> = if let Some(input) = conditioned {
            input.coordinates.iter().map(|&[t,h,w]| [t,h,w,0]).collect()
        } else {
            tokens.iter().enumerate().map(|(i,_)| [position + i as i32; 4]).collect()
        };
        rows.coordinates.write(&coordinates.iter().flatten().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        let mut overlays = Vec::new();
        if let Some(input) = conditioned {
            let mut compiler = PlanCompiler::new(&self.device, &self.program, self.settings.clone());
            let row_bytes = usize::try_from(self.geometry.hidden).map_err(|_| "hidden width overflow")?.checked_mul(4).ok_or("hidden row bytes overflow")?;
            for feature in &input.features {
                let offset = feature.destination.checked_mul(row_bytes).ok_or("feature destination overflow")?;
                let length = feature.count.checked_mul(row_bytes).ok_or("feature destination overflow")?;
                let out = rows.hidden.view(offset..offset.checked_add(length).ok_or("feature destination overflow")?)?;
                let composition = Composition::compile(&mut compiler, CompositionSpec {
                    entry: "cast_rows".into(),
                    shapes: shape(&[("M", feature.count as u64), ("K", self.geometry.hidden)])?,
                    elements: HashMap::from([("T".into(), Elem::Dtype(DType::F32)), ("U".into(), Elem::Dtype(DType::F32))]),
                    weights: HashMap::new(), external: names(&["input", "out"]), intermediates: HashSet::new(), scalars: HashMap::new(),
                })?;
                overlays.push((composition, HashMap::from([("input".into(), feature.source.clone()), ("out".into(), out)])));
            }
        }
        let ranges = if ranges.is_empty() {
            vec![(0, 0)]
        } else {
            ranges
        };
        rows.visible.write(
            &tokens
                .iter()
                .flat_map(|_| {
                    ranges
                        .iter()
                        .flat_map(|&(start, count)| [start as i32, (start + count) as i32])
                })
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        rows.tokens.write(
            &tokens
                .iter()
                .flat_map(|&t| (t as i32).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut advance = state.begin(tokens.len())?;
        advance.execute(|transition| {
            rows.destinations.write(
                &transition
                    .destinations
                    .iter()
                    .flat_map(|&d| (d as i32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            execute_stage(
                &mut rows.embedding,
                &HashMap::from([
                    ("out".into(), rows.hidden.clone()),
                    ("tokens".into(), rows.tokens.clone()),
                ]),
                &HashMap::new(),
                "embedding",
                None,
                &mut observations,
                &mut submission,
            )?;
            for (composition, tensors) in &mut overlays {
                execute_stage(composition, tensors, &HashMap::new(), "conditioning", None, &mut observations, &mut submission)?;
            }
            for (block_index, block) in rows.blocks.iter_mut().enumerate() {
                let mut tensors = HashMap::from([
                    ("hidden".into(), rows.hidden.clone()),
                    ("out".into(), rows.hidden.clone()),
                ]);
                let parameters = if block.attention {
                    let i = block.state_index;
                    tensors.extend([
                        ("destinations".into(), rows.destinations.clone()),
                        ("coordinates".into(), rows.coordinates.clone()),
                        ("visible".into(), rows.visible.clone()),
                        ("history_key".into(), transition.history[i].clone()),
                        ("history_value".into(), transition.history[i + 1].clone()),
                    ]);
                    HashMap::new()
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
                        ("residual".into(), rows.hidden.clone()),
                        ("out".into(), rows.hidden.clone()),
                    ]),
                    &HashMap::new(),
                    "feedforward",
                    Some(block_index),
                    &mut observations,
                    &mut submission,
                )?;
            }
            if let Some(selected) = selected.as_mut() {
                execute_stage(
                    &mut selected.readout,
                    &HashMap::from([
                        ("hidden".into(), rows.hidden.clone()),
                        ("selected".into(), selected.ids.clone()),
                        ("logits".into(), selected.logits.clone()),
                    ]),
                    &HashMap::new(),
                    "readout_selected",
                    None,
                    &mut observations,
                    &mut submission,
                )?;
            } else if !matches!(&readout, Readout::StateOnly | Readout::Selected(_)) {
                execute_stage(
                    &mut rows.readout,
                    &HashMap::from([
                        ("hidden".into(), rows.hidden.clone()),
                        ("logits".into(), rows.logits.clone()),
                    ]),
                    &HashMap::new(),
                    "readout",
                    None,
                    &mut observations,
                    &mut submission,
                )?;
            }
            if let Some(submission) = &mut submission {
                batch_observation = Some(submission.execute_batched()?);
            }
            Ok(())
        })?;
        let output = match readout {
            Readout::StateOnly => ReadoutOutput::StateOnly,
            Readout::Selected(_) => {
                let mut bytes = vec![0; selected.as_ref().map_or(0, |s| s.logits.len())];
                if let Some(selected) = selected {
                    selected.logits.read(&mut bytes)?;
                }
                ReadoutOutput::Selected(
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            }
            Readout::Logits => {
                let mut bytes = vec![0; rows.logits.len()];
                rows.logits.read(&mut bytes)?;
                ReadoutOutput::Logits(
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            }
            Readout::Sample {
                mask,
                sampling,
                seed,
                position,
            } => ReadoutOutput::Sample(self.sampler.as_mut().expect("sampler prepared").sample(
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
            batch_observation,
        ))
    }
}

#[cfg(test)]
mod conditioned_transaction_tests {
    use super::*;
    use super::super::{inputs::InputState, preparation::InputPlan};
    use crate::inputs::TokenId;
    #[test]
    fn conditioning_commits_only_after_numerical_completion_and_acceptance() {
        let device = Rc::new(Device::cpu());
        let store = StateStore::new(device.clone(), 8, 8, vec![], vec![]).unwrap();
        let mut state = store.create().unwrap();
        let mut input = InputState::new(&device, Rc::new(InputPlan::text(vec![TokenId(7)]).unwrap()), 0, vec![], 3).unwrap();
        let next = input.after(1).unwrap();
        let unfinished = ExecutedAdvance { advance: state.begin(1).unwrap(), output: ReadoutOutput::StateOnly };
        assert!(ConditionedAdvance { advance: unfinished, input: &mut input, next }.commit().is_err());
        assert_eq!((state.position(), input.position()), (0,0));
        for accept in [false, true] {
            let next = input.after(1).unwrap();
            let mut advance = state.begin(1).unwrap();
            // Lifecycle-only completion; this test makes no numerical claim.
            advance.execute(|_| Ok(())).unwrap();
            let staged = ConditionedAdvance { advance: ExecutedAdvance { advance, output: ReadoutOutput::StateOnly }, input: &mut input, next };
            if accept { staged.commit().unwrap(); } else { staged.abort(); }
            assert_eq!((state.position(), input.position()), if accept {(1,1)} else {(0,0)});
        }
    }
}
