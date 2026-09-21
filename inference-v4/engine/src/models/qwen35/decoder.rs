//! Qwen sequence forwards and multi-request generation. Sequence successors
//! remain private until shared completion and independent request acceptance.
//! Every numerical composition is prepared once, eagerly, under the decoder's
//! declared workload envelope; a forward selects prepared capacity classes
//! and never compiles.
mod conditioning;
mod packed;
use super::{Description, FeedForwardWeights, Geometry, HeadMapping, MixerWeights};
use crate::{
    execution::{self, StageBatch},
    generation::{
        sampling::{Sampler, Selection},
        Proposal, Sampling,
    },
    models::sequence::{Advance, OwnedSequence, SequenceWork},
    preparation::{
        CompositionSpec, EnvelopeShape, IntegerRange, PreparedComposition, PreparationSession,
        Settings, WorkloadEnvelope,
    },
    state::{ComponentSpec, SequenceState, StateAdvance, StateStore},
    weights::{descriptor::WeightDescriptor, residency::ResidentWeight},
    Error,
};
use conditioning::{Conditioning, PreparedOverlay};
use seismic_lang::types::{DType, Elem};
use seismic_runtime::{plan::InvocationResults, Buffer, Device, ExecutionObservation};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
};
fn names(xs: &[&str]) -> HashSet<String> {
    xs.iter().map(|s| (*s).into()).collect()
}
fn weights(xs: Vec<(&str, ResidentWeight)>) -> HashMap<String, ResidentWeight> {
    xs.into_iter().map(|(n, w)| (n.into(), w)).collect()
}
fn scalar(xs: &[(&str, f64)]) -> HashMap<String, f64> {
    xs.iter().map(|(n, v)| ((*n).into(), *v)).collect()
}
fn exact(value: u64) -> EnvelopeShape {
    EnvelopeShape::Exact(value)
}
fn varying(capacity: u64) -> EnvelopeShape {
    EnvelopeShape::Bounded {
        min: 1,
        max: capacity,
        expected: 1,
    }
}
/// The declared decoder workload envelope: what invocations the prepared
/// decoder must admit. Context capacity bounds forward rows and rotary
/// positions; `max_ranges` bounds attention visibility fragmentation;
/// `readout_capacity` bounds selected-row readouts. Packed row capacity is
/// `max_sequences * context_capacity`.
pub struct DecoderWorkload {
    pub context_capacity: usize,
    pub max_sequences: usize,
    pub max_ranges: usize,
    pub readout_capacity: usize,
}
struct Block {
    mixer: PreparedComposition,
    feedforward: PreparedComposition,
    state_index: usize,
    attention: bool,
}
struct Rows {
    hidden: Buffer,
    logits: Buffer,
    coordinates: Buffer,
    visible: Buffer,
    tokens: Buffer,
    destinations: Buffer,
}
struct SelectedRows {
    ids: Buffer,
    logits: Buffer,
}
pub struct Decoder {
    geometry: Geometry,
    store: Rc<StateStore>,
    device: Rc<Device>,
    conditioning: Conditioning,
    context_capacity: usize,
    max_ranges: usize,
    packed_rows: u64,
    embedding: PreparedComposition,
    blocks: Vec<Block>,
    readout: PreparedComposition,
    selected: PreparedComposition,
    sampler: Sampler,
    rows: HashMap<(usize, usize), Rows>,
    selected_rows: HashMap<usize, SelectedRows>,
    packed: HashMap<usize, packed::PackedBuffers>,
}
#[derive(Clone, Debug)]
pub struct DecoderStepObservation {
    pub stage: String,
    pub block: Option<usize>,
    pub entry: String,
    pub execution: ExecutionObservation,
}
struct ConditionedInput<'a> {
    coordinates: &'a [[i32; 4]],
    overlays: &'a [PreparedOverlay],
}
fn forward_shapes(count: usize) -> BTreeMap<String, u64> {
    BTreeMap::from([("M".into(), count as u64)])
}
fn attention_shapes(count: usize, ranges: usize) -> BTreeMap<String, u64> {
    BTreeMap::from([("M".into(), count as u64), ("R".into(), ranges as u64)])
}
fn selected_shapes(count: usize, ids: usize) -> BTreeMap<String, u64> {
    BTreeMap::from([("M".into(), count as u64), ("S".into(), ids as u64)])
}
fn row_hidden_bytes(geometry: &Geometry) -> Result<usize, Error> {
    usize::try_from(geometry.hidden)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "residual allocation overflow".into())
}
fn logits_bytes(geometry: &Geometry) -> Result<usize, Error> {
    usize::try_from(geometry.vocabulary)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "logits allocation overflow".into())
}
fn rows_entry<'a>(
    geometry: &Geometry,
    device: &Rc<Device>,
    rows: &'a mut HashMap<(usize, usize), Rows>,
    count: usize,
    ranges: usize,
    context_capacity: usize,
) -> Result<&'a mut Rows, Error> {
    if count == 0 || ranges == 0 || count > context_capacity {
        return Err("invalid forward row count".into());
    }
    if !rows.contains_key(&(count, ranges)) {
        let bytes = |width: usize| count.checked_mul(width).ok_or("forward buffer overflow");
        let visible_width = ranges
            .checked_mul(8)
            .ok_or("visibility buffer overflow")?;
        let allocated = Rows {
            hidden: device.buffer(bytes(row_hidden_bytes(geometry)?)?)?,
            logits: device.buffer(logits_bytes(geometry)?)?,
            coordinates: device.buffer(bytes(16)?)?,
            visible: device.buffer(bytes(visible_width)?)?,
            tokens: device.buffer(bytes(4)?)?,
            destinations: device.buffer(bytes(4)?)?,
        };
        rows.insert((count, ranges), allocated);
    }
    Ok(rows.get_mut(&(count, ranges)).expect("row buffers prepared"))
}
fn selected_entry<'a>(
    device: &Rc<Device>,
    context_capacity: usize,
    selected: &'a mut HashMap<usize, SelectedRows>,
    ids: usize,
) -> Result<&'a mut SelectedRows, Error> {
    if ids == 0 || ids > context_capacity {
        return Err("selected readout size exceeds the prepared envelope".into());
    }
    if !selected.contains_key(&ids) {
        let bytes = ids
            .checked_mul(4)
            .ok_or("selected readout size overflow")?;
        selected.insert(
            ids,
            SelectedRows {
                ids: device.buffer(bytes)?,
                logits: device.buffer(bytes)?,
            },
        );
    }
    Ok(selected.get_mut(&ids).expect("selection prepared"))
}
fn execute_stage(
    composition: &PreparedComposition,
    shapes: &BTreeMap<String, u64>,
    tensors: &HashMap<String, Buffer>,
    stage: &str,
    block: Option<usize>,
    observations: &mut Option<Vec<DecoderStepObservation>>,
    batch: &mut Option<StageBatch>,
) -> Result<InvocationResults, Error> {
    let invocation = execution::invoke(composition, shapes, tensors, &HashMap::new())?;
    let results = execution::result_planes(&invocation);
    let entry = composition.entry().to_string();
    if let Some(batch) = batch {
        batch.add(invocation);
    } else if let Some(observations) = observations {
        let (_, observation) = execution::run_observed(invocation)?;
        observations.push(DecoderStepObservation {
            stage: stage.into(),
            block,
            entry,
            execution: observation,
        });
    } else {
        execution::run(invocation)?;
    }
    Ok(results)
}
fn result_buffer(results: &InvocationResults, path: &[u32]) -> Result<Buffer, Error> {
    execution::result_buffer(results, path).map_err(Error::from)
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
                let tokens: Vec<_> = row.proposal.tokens().iter().map(|token| token.0).collect();
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
    /// Prepare the whole decoder under one declared workload envelope. Every
    /// capacity class of every composition is compiled and natively sealed
    /// before the decoder is returned; a failure retains nothing.
    pub fn compile(
        device: Rc<Device>,
        description: &Description,
        mut import: impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, String>,
        settings: Settings,
        workload: DecoderWorkload,
    ) -> Result<Self, Error> {
        let g = &description.geometry;
        g.validate().map_err(|e| e.to_string())?;
        if description.blocks.len() != g.layers.len() {
            return Err("decoder requires complete block descriptors".into());
        }
        let DecoderWorkload {
            context_capacity,
            max_sequences,
            max_ranges,
            readout_capacity,
        } = workload;
        if context_capacity == 0
            || context_capacity as u64 > g.context_limit
            || context_capacity > i32::MAX as usize
            || max_sequences == 0
            || max_ranges == 0
            || max_ranges > context_capacity
            || readout_capacity == 0
            || readout_capacity > context_capacity
        {
            return Err("invalid decoder workload envelope".into());
        }
        let history_capacity = context_capacity
            .checked_mul(max_sequences)
            .filter(|n| *n <= i32::MAX as usize)
            .ok_or("history capacity overflow")?;
        let packed_rows = u64::try_from(
            max_sequences
                .checked_mul(context_capacity)
                .ok_or("packed capacity overflow")?,
        )
        .ok()
        .filter(|&n| n <= i64::from(i32::MAX) as u64)
        .ok_or("packed capacity exceeds index domain")?;
        let program = super::program::program()?;
        let mut session = PreparationSession::new(&device, &program, settings.clone());
        let activation = g.activation_dtype;
        let elements = BTreeMap::from([("A".into(), Elem::Dtype(activation))]);
        let mut bound = |entry: &str,
                         shapes: BTreeMap<String, EnvelopeShape>,
                         bound_weights: HashMap<String, ResidentWeight>,
                         external: &[&str],
                         bound_scalars: HashMap<String, f64>| {
            session.prepare(CompositionSpec {
                entry: entry.into(),
                envelope: WorkloadEnvelope::geometric(shapes, elements.clone())?,
                weights: bound_weights,
                external: names(external),
                intermediates: HashSet::new(),
                scalars: bound_scalars,
            })
        };
        let embedding = bound(
            "qwen_embedding_rows",
            BTreeMap::from([
                ("M".into(), varying(packed_rows)),
                ("V".into(), exact(g.vocabulary)),
                ("D".into(), exact(g.hidden)),
            ]),
            weights(vec![("table", import(&description.embedding, activation)?)]),
            &["tokens"],
            HashMap::new(),
        )?
        .control_domain(
            "tokens",
            IntegerRange {
                min: 0,
                max: i128::from(g.vocabulary) - 1,
            },
        )?;
        let attention_envelope = BTreeMap::from([
            ("M".into(), varying(context_capacity as u64)),
            ("D".into(), exact(g.hidden)),
            ("T".into(), exact(history_capacity as u64)),
            ("R".into(), varying(max_ranges as u64)),
            ("G".into(), exact(g.attention_heads / g.kv_heads)),
            ("KV".into(), exact(g.kv_heads)),
            ("P".into(), exact(g.rotary_width / 2)),
            (
                "S".into(),
                exact(g.attention_width - g.rotary_width),
            ),
            ("SH".into(), exact(g.rotary_sections[1])),
            ("SW".into(), exact(g.rotary_sections[2])),
        ]);
        let recurrent_envelope = BTreeMap::from([
            ("M".into(), varying(context_capacity as u64)),
            ("H".into(), exact(g.hidden)),
            ("NK".into(), exact(g.recurrent_key_heads)),
            (
                "GV".into(),
                exact(g.recurrent_value_heads / g.recurrent_key_heads),
            ),
            ("W".into(), exact(g.recurrent_width)),
            ("C".into(), exact(g.convolution_width)),
        ]);
        let dense_envelope = BTreeMap::from([
            ("M".into(), varying(context_capacity as u64)),
            ("H".into(), exact(g.hidden)),
            ("F".into(), exact(g.intermediate)),
        ]);
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
                        attention_envelope.clone(),
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
                            "destinations",
                            "coordinates",
                            "visible",
                            "history_key",
                            "history_value",
                        ],
                        scalar(&[
                            ("base", g.rotary_base),
                            ("epsilon", g.epsilon),
                            ("scale", 1.0 / (g.attention_width as f64).sqrt()),
                        ]),
                    )?
                    .control_inputs(&["visible"])?
                    .control_domain(
                        "coordinates",
                        IntegerRange {
                            min: 0,
                            max: i128::from(i32::MAX),
                        },
                    )?
                    .control_domain(
                        "destinations",
                        IntegerRange {
                            min: 0,
                            max: history_capacity as i128 - 1,
                        },
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
                        "qwen_recurrent_sequence",
                        recurrent_envelope.clone(),
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
                        &["hidden", "window", "delta"],
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
                        dense_envelope.clone(),
                        weights(vec![
                            ("norm", import(&block.feedforward_norm, activation)?),
                            ("gate_weight", import(&ff.gate, activation)?),
                            ("up_weight", import(&ff.up, activation)?),
                            ("down_weight", import(&ff.down, activation)?),
                        ]),
                        &["residual"],
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
                        BTreeMap::from([
                            ("M".into(), varying(context_capacity as u64)),
                            ("H".into(), exact(g.hidden)),
                            ("E".into(), exact(experts.count)),
                            ("K".into(), exact(experts.selected)),
                            ("F".into(), exact(experts.intermediate)),
                            ("S".into(), exact(experts.shared_intermediate)),
                        ]),
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
                        &["residual"],
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
        let selected = bound(
            "qwen_readout_selected",
            BTreeMap::from([
                ("M".into(), varying(context_capacity as u64)),
                ("V".into(), exact(g.vocabulary)),
                ("D".into(), exact(g.hidden)),
                ("S".into(), varying(readout_capacity as u64)),
            ]),
            readout_weights.clone(),
            &["hidden", "selected"],
            scalar(&[("epsilon", g.epsilon)]),
        )?
        .control_domain(
            "selected",
            IntegerRange {
                min: 0,
                max: i128::from(g.vocabulary) - 1,
            },
        )?;
        let readout = bound(
            "qwen_readout_rows",
            BTreeMap::from([
                ("M".into(), varying(context_capacity as u64)),
                ("V".into(), exact(g.vocabulary)),
                ("D".into(), exact(g.hidden)),
            ]),
            readout_weights,
            &["hidden"],
            scalar(&[("epsilon", g.epsilon)]),
        )?;
        let store = StateStore::new(
            device.clone(),
            context_capacity,
            history_capacity,
            history_rows,
            components,
        )?;
        let vocabulary = usize::try_from(g.vocabulary)
            .map_err(|_| "vocabulary exceeds the host index domain")?;
        let sampler = Sampler::compile(&device, vocabulary, settings.clone())?;
        let conditioning = Conditioning::new(device.clone(), program, settings, g.hidden);
        let mut rows = HashMap::new();
        rows_entry(g, &device, &mut rows, 1, 1, context_capacity)?;
        Ok(Self {
            geometry: g.clone(),
            store,
            device,
            conditioning,
            context_capacity,
            max_ranges,
            packed_rows,
            embedding,
            blocks,
            readout,
            selected,
            sampler,
            rows,
            selected_rows: HashMap::new(),
            packed: HashMap::new(),
        })
    }
    fn rows_for(&mut self, count: usize, ranges: usize) -> Result<&mut Rows, Error> {
        let Decoder {
            geometry,
            device,
            rows,
            context_capacity,
            ..
        } = self;
        rows_entry(geometry, device, rows, count, ranges, *context_capacity)
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
    /// Drop idle row buffers and selection storage. Prepared compositions are
    /// retained; preparation state never depends on execution.
    pub fn reclaim_idle(&mut self) -> Result<usize, Error> {
        let before = self.device.memory_usage().charged;
        self.rows.retain(|geometry, _| *geometry == (1, 1));
        self.selected_rows.clear();
        self.packed.clear();
        self.store.release_idle()?;
        Ok(before - self.device.memory_usage().charged)
    }
    pub fn state_store(&self) -> &Rc<StateStore> {
        &self.store
    }
    /// Kernels compiled for the prepared decoder. Constant after preparation:
    /// a forward can never compile, so a nonzero difference is impossible.
    pub fn compiled_kernel_count(&self) -> usize {
        self.embedding.kernel_count()
            + self
                .blocks
                .iter()
                .map(|b| b.mixer.kernel_count() + b.feedforward.kernel_count())
                .sum::<usize>()
            + self.readout.kernel_count()
            + self.selected.kernel_count()
            + self.sampler.kernel_count()
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
        // Explicit conditioned-input preparation: every overlay composition is
        // compiled here, in full, before any numerical inference submission is
        // built by the forward below.
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
        let (advance, _, _) =
            self.propose_impl(state, tokens, false, true, readout, Some(&conditioned))?;
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
        observed: bool,
        batched: bool,
        readout: Readout<'_>,
        conditioned: Option<&ConditionedInput<'_>>,
    ) -> Result<
        (
            ExecutedAdvance<'a>,
            Vec<DecoderStepObservation>,
            Option<ExecutionObservation>,
        ),
        Error,
    > {
        let mut observations = observed.then(Vec::new);
        let mut batch = batched.then(StageBatch::default);
        let mut batch_observation = None;
        let Decoder {
            geometry,
            store,
            device,
            context_capacity,
            max_ranges,
            embedding,
            blocks,
            readout: readout_stage,
            selected: selected_stage,
            sampler,
            rows,
            selected_rows,
            ..
        } = self;
        if !state.belongs_to(store) {
            return Err("sequence belongs to another decoder state store".into());
        }
        if tokens.is_empty()
            || tokens
                .iter()
                .any(|&t| u64::from(t) >= geometry.vocabulary || t > i32::MAX as u32)
        {
            return Err("token is outside vocabulary".into());
        }
        let ranges = state.history_ranges();
        if tokens.len() > *context_capacity - state.position() {
            return Err("forward exceeds context limit".into());
        }
        if ranges.len() > *max_ranges {
            return Err("forward visibility exceeds the prepared envelope".into());
        }
        let selected_ids = match &readout {
            Readout::Selected(ids) => {
                if ids
                    .iter()
                    .any(|&id| u64::from(id) >= geometry.vocabulary || id > i32::MAX as u32)
                {
                    return Err("selected readout token is outside vocabulary".into());
                }
                Some(ids.len())
            }
            _ => None,
        };
        let shape_ranges = ranges.len().max(1);
        let mut selected = selected_ids
            .filter(|&ids| ids > 0)
            .map(|ids| selected_entry(device, *context_capacity, selected_rows, ids))
            .transpose()?;
        if let (Some(entry), Readout::Selected(ids)) = (selected.as_ref(), &readout) {
            entry.ids.write(
                &ids.iter()
                    .flat_map(|&id| (id as i32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        let rows =
            rows_entry(geometry, device, rows, tokens.len(), shape_ranges, *context_capacity)?;
        let position =
            i32::try_from(state.position()).map_err(|_| "rotary position exceeds index domain")?;
        let coordinates: Vec<[i32; 4]> = match conditioned {
            Some(input) => input.coordinates.to_vec(),
            None => tokens
                .iter()
                .enumerate()
                .map(|(i, _)| [position + i as i32; 4])
                .collect(),
        };
        rows.coordinates.write(
            &coordinates
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
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
        let mut state_results = Vec::new();
        advance.execute(|transition| {
            rows.destinations.write(
                &transition
                    .destinations
                    .iter()
                    .flat_map(|&d| (d as i32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let embedding = execute_stage(
                embedding,
                &forward_shapes(tokens.len()),
                &HashMap::from([("tokens".into(), rows.tokens.clone())]),
                "embedding",
                None,
                &mut observations,
                &mut batch,
            )?;
            let mut hidden = result_buffer(&embedding, &[1])?;
            if let Some(input) = conditioned {
                for overlay in input.overlays {
                    let out = hidden.view(
                        overlay.offset
                            ..overlay
                                .offset
                                .checked_add(overlay.length)
                                .ok_or("feature destination overflow")?,
                    )?;
                    execute_stage(
                        &overlay.composition,
                        &forward_shapes(overlay.count),
                        &HashMap::from([
                            ("input".into(), overlay.source.clone()),
                            ("out".into(), out),
                        ]),
                        "conditioning",
                        None,
                        &mut observations,
                        &mut batch,
                    )?;
                }
            }
            for (block_index, block) in blocks.iter().enumerate() {
                let mut tensors = HashMap::from([("hidden".into(), hidden.clone())]);
                if block.attention {
                    let i = block.state_index;
                    tensors.extend([
                        ("destinations".into(), rows.destinations.clone()),
                        ("coordinates".into(), rows.coordinates.clone()),
                        ("visible".into(), rows.visible.clone()),
                        ("history_key".into(), transition.history[i].clone()),
                        ("history_value".into(), transition.history[i + 1].clone()),
                    ]);
                } else {
                    let i = block.state_index;
                    tensors.extend([
                        ("window".into(), transition.previous[i].clone()),
                        ("delta".into(), transition.previous[i + 1].clone()),
                    ]);
                }
                let shapes = if block.attention {
                    attention_shapes(tokens.len(), shape_ranges)
                } else {
                    forward_shapes(tokens.len())
                };
                let mixed = execute_stage(
                    &block.mixer,
                    &shapes,
                    &tensors,
                    "mixer",
                    Some(block_index),
                    &mut observations,
                    &mut batch,
                )?;
                hidden = if block.attention {
                    result_buffer(&mixed, &[])?
                } else {
                    let i = block.state_index;
                    state_results.push((i, result_buffer(&mixed, &[0])?));
                    state_results.push((i + 1, result_buffer(&mixed, &[1])?));
                    result_buffer(&mixed, &[2])?
                };
                let feedforward = execute_stage(
                    &block.feedforward,
                    &forward_shapes(tokens.len()),
                    &HashMap::from([("residual".into(), hidden.clone())]),
                    "feedforward",
                    Some(block_index),
                    &mut observations,
                    &mut batch,
                )?;
                hidden = result_buffer(&feedforward, &[6])
                    .or_else(|_| result_buffer(&feedforward, &[14]))?;
            }
            if let (Some(selected), Some(ids)) = (selected.as_mut(), selected_ids) {
                let results = execute_stage(
                    selected_stage,
                    &selected_shapes(tokens.len(), ids),
                    &HashMap::from([
                        ("hidden".into(), hidden.clone()),
                        ("selected".into(), selected.ids.clone()),
                    ]),
                    "readout_selected",
                    None,
                    &mut observations,
                    &mut batch,
                )?;
                selected.logits = result_buffer(&results, &[1])?;
            } else if !matches!(&readout, Readout::StateOnly | Readout::Selected(_)) {
                let results = execute_stage(
                    readout_stage,
                    &forward_shapes(tokens.len()),
                    &HashMap::from([("hidden".into(), hidden.clone())]),
                    "readout",
                    None,
                    &mut observations,
                    &mut batch,
                )?;
                rows.logits = result_buffer(&results, &[1])?;
            }
            if let Some(batch) = batch.take() {
                let observed = batch.execute_observed()?;
                batch_observation = Some(execution::batch_observation(&observed));
            }
            Ok(())
        })?;
        for (index, buffer) in state_results {
            advance.replace_following(index, buffer)?;
        }
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
            } => ReadoutOutput::Sample(
                sampler.sample(&rows.logits, mask, sampling, seed, position)?,
            ),
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
    use super::super::{inputs::InputState, preparation::InputPlan};
    use super::*;
    use crate::inputs::TokenId;
    #[test]
    #[ignore = "requires a Metal device"]
    fn conditioning_commits_only_after_numerical_completion_and_acceptance() {
        let device = Rc::new(Device::metal().unwrap());
        let store = StateStore::new(device.clone(), 8, 8, vec![], vec![]).unwrap();
        let mut state = store.create().unwrap();
        let mut input = InputState::new(
            &device,
            Rc::new(InputPlan::text(vec![TokenId(7)]).unwrap()),
            0,
            vec![],
            3,
        )
        .unwrap();
        let next = input.after(1).unwrap();
        let unfinished = ExecutedAdvance {
            advance: state.begin(1).unwrap(),
            output: ReadoutOutput::StateOnly,
        };
        assert!(ConditionedAdvance {
            advance: unfinished,
            input: &mut input,
            next
        }
        .commit()
        .is_err());
        assert_eq!((state.position(), input.position()), (0, 0));
        for accept in [false, true] {
            let next = input.after(1).unwrap();
            let mut advance = state.begin(1).unwrap();
            // Lifecycle-only completion; this test makes no numerical claim.
            advance.execute(|_| Ok(())).unwrap();
            let staged = ConditionedAdvance {
                advance: ExecutedAdvance {
                    advance,
                    output: ReadoutOutput::StateOnly,
                },
                input: &mut input,
                next,
            };
            if accept {
                staged.commit().unwrap();
            } else {
                staged.abort();
            }
            assert_eq!(
                (state.position(), input.position()),
                if accept { (1, 1) } else { (0, 0) }
            );
        }
    }
}
