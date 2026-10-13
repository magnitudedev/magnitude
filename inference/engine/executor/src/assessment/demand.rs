//! Header-derived decode demand: the bytes one plain target decode step
//! streams, the history it reads per token of context, and the entry calls
//! it launches. Derived from the same program and load plans native
//! preparation uses, and from the state layout the memory terms use for
//! history bytes.

use super::AssessmentError;
use crate::operators::{InputForm, parallel::DenseBesideRouted, streams::StreamsBinding};
use crate::{
    FeedForwardProgramSlot, GeneralRoutedBinding, MixerProgramSlot, ModelLoadPlan,
    PerLayerEntryBinding, ReadoutHead, SublayerTail, WeightPlan,
};
use magnitude_family_contracts::{
    InputNorm, ModelDefinition, SublayerIndex, WeightKind, WeightScope,
};
use magnitude_state::{HistoryDomainLayout, KvCodec, LayerRef, ModelStateLayout};
use seismic::Element;

/// History one decode step reads for each token of context, summed over the
/// attention layers that share a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryRead {
    pub bytes_per_token: u64,
    /// The most tokens read: a window domain's window, or the blocks a
    /// selected read reads and its tail. `None` reads every token.
    pub window: Option<u64>,
}

impl HistoryRead {
    /// The bytes read at `depth` tokens of context.
    pub fn bytes_at(&self, depth: u32) -> u64 {
        let tokens = self
            .window
            .map_or(u64::from(depth), |window| window.min(u64::from(depth)));
        self.bytes_per_token.saturating_mul(tokens)
    }
}

/// One plain target decode step of one row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeDemand {
    /// Bytes streamed independent of context depth: weights (a routed
    /// layer's selected experts only), norms, routers, recurrent and
    /// state-space state, rows, logits and per-layer table uploads.
    pub streamed_bytes: u64,
    /// History read per token of context.
    pub history: Vec<HistoryRead>,
    /// Entry calls.
    pub launches: u64,
}

fn demand_error(message: impl Into<String>) -> AssessmentError {
    AssessmentError::Demand(message.into())
}

/// The output rows of a weight matrix (every extent but the reduction; an
/// expert tensor's rows over all its experts).
fn output_rows(weight: &WeightPlan) -> Result<u64, AssessmentError> {
    let [leading @ .., _] = weight.shape.as_slice() else {
        return Err(demand_error(format!(
            "weight {:?} has no extents",
            weight.descriptor.name
        )));
    };
    leading
        .iter()
        .try_fold(1u64, |rows, extent| rows.checked_mul(*extent))
        .ok_or_else(|| demand_error("weight rows overflow"))
}

/// The bytes a decode row streams of `selected` of an expert tensor's
/// `experts` equal slabs.
fn selected_bytes(
    weight: &WeightPlan,
    experts: u64,
    selected: u64,
) -> Result<u64, AssessmentError> {
    let rows = output_rows(weight)?;
    if experts == 0
        || selected > experts
        || weight.resident_bytes % experts != 0
        || rows % experts != 0
    {
        return Err(demand_error(format!(
            "expert tensor {:?} does not split into {experts} slabs",
            weight.descriptor.name
        )));
    }
    (weight.resident_bytes / experts)
        .checked_mul(selected)
        .ok_or_else(|| demand_error("selected expert streaming overflows"))
}

/// The step as it accumulates.
#[derive(Default)]
struct Step {
    streamed_bytes: u64,
    history: Vec<HistoryRead>,
    launches: u64,
}

impl Step {
    /// One entry call streaming `bytes`.
    fn launch(&mut self, bytes: u64) -> Result<(), AssessmentError> {
        self.launches = self
            .launches
            .checked_add(1)
            .ok_or_else(|| demand_error("entry call count overflows"))?;
        self.streamed_bytes = self
            .streamed_bytes
            .checked_add(bytes)
            .ok_or_else(|| demand_error("streamed bytes overflow"))?;
        Ok(())
    }

    /// One entry call streaming every weight in `weights` whole.
    fn project(&mut self, weights: &[&WeightPlan]) -> Result<(), AssessmentError> {
        let bytes = weights
            .iter()
            .try_fold(0u64, |total, weight| {
                total.checked_add(weight.resident_bytes)
            })
            .ok_or_else(|| demand_error("projection bytes overflow"))?;
        self.launch(bytes)
    }

    /// One attention entry call reading `bytes_per_token` of history per
    /// token of context, up to `window` tokens.
    fn attend(&mut self, bytes_per_token: u64, window: Option<u64>) -> Result<(), AssessmentError> {
        self.launch(0)?;
        self.read(bytes_per_token, window)
    }

    /// `bytes_per_token` of history read per token of context, up to
    /// `window` tokens, by an entry call already counted.
    fn read(&mut self, bytes_per_token: u64, window: Option<u64>) -> Result<(), AssessmentError> {
        match self.history.iter_mut().find(|read| read.window == window) {
            Some(read) => {
                read.bytes_per_token = read
                    .bytes_per_token
                    .checked_add(bytes_per_token)
                    .ok_or_else(|| demand_error("history bytes overflow"))?;
            }
            None => self.history.push(HistoryRead {
                bytes_per_token,
                window,
            }),
        }
        Ok(())
    }
}

/// The planned weight of `kind` in `scope`, checked against the element the
/// program slot binds.
fn planned<'a>(
    load: &'a ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    bound: Element,
) -> Result<&'a WeightPlan, AssessmentError> {
    let weight = load
        .target()
        .iter()
        .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
        // An output projection of a scope stored as planes is planned and
        // not placed: its canonical full-projection demand stands.
        .or_else(|| {
            (kind == WeightKind::Output)
                .then(|| load.planned_output(scope))
                .flatten()
        })
        .ok_or_else(|| demand_error(format!("{kind:?} weight is absent for {scope:?}")))?;
    if weight.resident != bound {
        return Err(demand_error(format!(
            "{kind:?} binding for {scope:?} disagrees with the resident load plan"
        )));
    }
    Ok(weight)
}

/// A sublayer's output by its tail: the operator's own output entry
/// (`residual`), or the `output` projection into F32 rows and the post-norm
/// row op, whose norm weight `planned` checks.
fn tail<'a>(
    step: &mut Step,
    tail: SublayerTail,
    output: &WeightPlan,
    planned: impl Fn(WeightKind, Element) -> Result<&'a WeightPlan, AssessmentError>,
) -> Result<(), AssessmentError> {
    match tail {
        SublayerTail::Residual => step.project(&[output]),
        SublayerTail::PostNorm { norm, .. } => {
            post_norm(step, output, planned(WeightKind::PostNorm, norm)?)
        }
    }
}

/// A post-norm tail: the `output` projection into F32 rows, then the row op
/// normalizing it with `norm` into the residual.
fn post_norm(
    step: &mut Step,
    output: &WeightPlan,
    norm: &WeightPlan,
) -> Result<(), AssessmentError> {
    step.project(&[output])?;
    step.launch(norm.resident_bytes)
}

/// The bytes of `elements` values of `element`.
fn row_bytes(element: Element, elements: u64) -> Result<u64, AssessmentError> {
    element
        .canonical_byte_len(&[elements])
        .map_err(|error| demand_error(format!("{} row of {elements}: {error}", element.name())))
}

/// `programs::graph::per_layer::per_layer_entry`, once per step after the
/// embedding: the row's host-table row gathered and uploaded, the embedded
/// row rounded and projected to every layer's channels, the uploaded row
/// converted to its resident representation, both combined, and the result
/// copied into the program's per-layer rows.
fn per_layer_entry(
    step: &mut Step,
    load: &ModelLoadPlan,
    entry: PerLayerEntryBinding,
) -> Result<(), AssessmentError> {
    let channels = entry
        .layers
        .checked_mul(entry.width)
        .ok_or_else(|| demand_error("per-layer channels overflow"))?;
    let table = load
        .host_tables()
        .iter()
        .find(|table| table.role.kind == WeightKind::PerLayerTable)
        .ok_or_else(|| demand_error("a per-layer entry without its host table"))?;
    let rows = table.shape[0];
    if table.source != entry.table_source || rows == 0 || table.bytes % rows != 0 {
        return Err(demand_error(format!(
            "host table {:?} disagrees with the per-layer entry",
            table.descriptor.name
        )));
    }
    step.launch(table.bytes / rows)?;
    step.launch(row_bytes(Element::f32(), entry.hidden)?)?;
    let projection = planned(
        load,
        WeightScope::Target,
        WeightKind::PerLayerModelProjection,
        entry.projection,
    )?;
    step.project(&[projection])?;
    step.launch(row_bytes(entry.table_source, channels)?)?;
    planned(
        load,
        WeightScope::Target,
        WeightKind::PerLayerProjectionNorm,
        entry.norm,
    )?;
    let f32_rows = row_bytes(Element::f32(), channels)?;
    step.launch(f32_rows)?;
    step.launch(f32_rows)
}

/// `operators::streams::graph::mix_site` in `scope`: `stream_mix_down`
/// streams the site's norm and down weights (and, `injected`, its inject
/// weights), and `stream_mix_up` its up weights and norm. The consumer entry
/// sums the partials in its prologue, and the sublayer's output entry
/// scatters into the streams in its epilogue: neither launches.
fn mix_site(
    step: &mut Step,
    load: &ModelLoadPlan,
    scope: WeightScope,
    streams: StreamsBinding,
    injected: bool,
) -> Result<(), AssessmentError> {
    let weight = |kind, bound| planned(load, scope, kind, bound);
    let norm = weight(WeightKind::HyperConnectionNorm, streams.norm)?;
    let mut down = vec![norm, weight(WeightKind::HyperConnectionDown, streams.down)?];
    if injected {
        down.push(weight(WeightKind::HyperConnectionInject, streams.inject)?);
    }
    step.project(&down)?;
    step.project(&[weight(WeightKind::HyperConnectionUp, streams.up)?, norm])
}

/// The input of the sublayer at `scope` as it runs: the residual under its
/// RMS input norm, which the consumer entry fuses (bound at `norm`), or the
/// mix site of a hyper-connection sublayer.
fn sublayer_input(
    step: &mut Step,
    load: &ModelLoadPlan,
    definition: &ModelDefinition,
    scope: WeightScope,
    norm: Element,
    streams: Option<StreamsBinding>,
) -> Result<(), AssessmentError> {
    let WeightScope::TargetSublayer(index) = scope else {
        return Err(demand_error(format!("{scope:?} is not a target sublayer")));
    };
    let sublayer = definition
        .decoder
        .blocks
        .get(index.block as usize)
        .and_then(|block| block.sublayers.get(index.sublayer as usize))
        .ok_or_else(|| demand_error(format!("{scope:?} is not a decoder sublayer")))?;
    match &sublayer.input {
        InputNorm::HyperConnectionMix(_) => {
            let streams =
                streams.ok_or_else(|| demand_error(format!("{scope:?} mixes without streams")))?;
            mix_site(step, load, scope, streams, true)
        }
        _ => planned(load, scope, WeightKind::InputNorm, norm).map(|_| ()),
    }
}

/// `operators::ngram` in `scope`, whose bank components (its window, then
/// its token window) start at `component`: the row rule over the token
/// window (`ngram_rows`), the read of its heads' stored rows from the
/// table's slots (`stored_rows`), the key and value projections, the gate
/// over the key and query norms (`ngram_gate`), and the row op over the
/// convolution norm, the taps and the window (`ngram_inject`).
fn ngram(
    step: &mut Step,
    load: &ModelLoadPlan,
    layout: &ModelStateLayout,
    scope: WeightScope,
    binding: crate::NGramBinding,
    component: usize,
    index: usize,
) -> Result<(), AssessmentError> {
    let weight = |kind, bound| planned(load, scope, kind, bound);
    step.launch(bank_bytes(layout, component + 1, 1, index)?)?;
    let table = weight(WeightKind::NGramTable, binding.table_source)?;
    let slots = table.shape[0];
    if slots == 0 || table.resident_bytes % slots != 0 {
        return Err(demand_error(format!(
            "n-gram table {:?} does not split into its slots",
            table.descriptor.name
        )));
    }
    step.launch(
        (table.resident_bytes / slots)
            .checked_mul(binding.heads())
            .ok_or_else(|| demand_error("n-gram row bytes overflow"))?,
    )?;
    step.project(&[weight(WeightKind::NGramKey, binding.key)?])?;
    step.project(&[weight(WeightKind::NGramValue, binding.value)?])?;
    step.launch(
        weight(WeightKind::NGramKeyNorm, binding.norm)?.resident_bytes
            + weight(WeightKind::NGramQueryNorm, binding.norm)?.resident_bytes,
    )?;
    step.launch(
        weight(WeightKind::NGramConvolutionNorm, binding.norm)?.resident_bytes
            + weight(WeightKind::NGramConvolution, Element::f32())?.resident_bytes
            + bank_bytes(layout, component, 1, index)?,
    )
}

/// `operators::routed` in `scope`: the shared expert onto the sum's root,
/// selection, the latent projections, and the selected experts' expansion
/// and down projection. Its input is the caller's.
fn general_routed(
    step: &mut Step,
    load: &ModelLoadPlan,
    scope: WeightScope,
    binding: &GeneralRoutedBinding,
) -> Result<(), AssessmentError> {
    let weight = |kind, bound| planned(load, scope, kind, bound);
    let shape = binding.shape;
    if let Some((gate, up, down)) = binding.shared {
        let up = weight(WeightKind::SharedUp, up)?;
        match gate {
            Some(gate) => step.project(&[weight(WeightKind::SharedGate, gate)?, up])?,
            None => step.project(&[up])?,
        }
        step.project(&[weight(WeightKind::SharedDown, down)?])?;
    }
    step.launch(weight(WeightKind::Router, binding.router)?.resident_bytes)?;
    if let Some((down, up)) = binding.latent {
        step.project(&[weight(WeightKind::LatentDown, down)?])?;
        step.project(&[weight(WeightKind::LatentUp, up)?])?;
    }
    let chosen = |weight: &WeightPlan| selected_bytes(weight, shape.experts, shape.selected);
    let mut expansion = chosen(weight(WeightKind::ExpertUp, binding.expert_up)?)?;
    if let Some(gate) = binding.expert_gate {
        expansion = expansion
            .checked_add(chosen(weight(WeightKind::ExpertGate, gate)?)?)
            .ok_or_else(|| demand_error("expert expansion bytes overflow"))?;
    }
    step.launch(expansion)?;
    step.launch(chosen(weight(
        WeightKind::ExpertDown,
        binding.expert_down,
    )?)?)
}

fn usize_bytes(bytes: usize) -> Result<u64, AssessmentError> {
    u64::try_from(bytes).map_err(|_| demand_error("state bytes exceed u64"))
}

/// What one decode row of `layer`'s attention reads: its history's bytes per
/// token and, for a window domain, the most tokens it reads. A layer sharing
/// another layer's history reads the source's.
fn history_read(
    domains: &[HistoryDomainLayout],
    layer: LayerRef,
) -> Result<(u64, Option<u64>), AssessmentError> {
    for domain in domains {
        let window = match domain {
            HistoryDomainLayout::Token { .. } => None,
            // A window domain's rows are its window's tokens.
            HistoryDomainLayout::Window { rows, .. } => Some(usize_bytes(*rows)?),
            // An indexer's block keys: no attention layer's history.
            HistoryDomainLayout::Block { .. } => continue,
            HistoryDomainLayout::Shared { source, layers } => {
                if layers.contains(&layer) {
                    return history_read(domains, *source);
                }
                continue;
            }
        };
        if let Some(row_bytes) = component_row_bytes(domain, layer)? {
            return Ok((row_bytes, window));
        }
    }
    Err(demand_error(format!(
        "attention layer {layer:?} has no history"
    )))
}

/// The bytes of one row of `layer`'s component in `domain`, when it has one.
fn component_row_bytes(
    domain: &HistoryDomainLayout,
    layer: LayerRef,
) -> Result<Option<u64>, AssessmentError> {
    domain
        .components()
        .iter()
        .find(|component| component.layer == layer)
        .map(|component| {
            component.planes().iter().try_fold(0u64, |total, plane| {
                total
                    .checked_add(usize_bytes(plane.row_bytes)?)
                    .ok_or_else(|| demand_error("history row bytes overflow"))
            })
        })
        .transpose()
}

/// One decode row's `indexer_select` call for target block `layer`'s
/// selected read of `shape`: it reads the raw keys of the positions before
/// it in its block and writes its own and the block it completes, then
/// scores every complete block (reading its key, writing and reading its F32
/// score) and selects. Per token of context that is one block's key and
/// score traffic over the block's rows.
fn indexer_select(
    step: &mut Step,
    domains: &[HistoryDomainLayout],
    layer: u32,
    shape: crate::SelectionShape,
) -> Result<(), AssessmentError> {
    let indexer = LayerRef::Indexer(layer);
    let mut raw = None;
    let mut finished = None;
    for domain in domains {
        match domain {
            HistoryDomainLayout::Window { .. } if raw.is_none() => {
                raw = component_row_bytes(domain, indexer)?;
            }
            HistoryDomainLayout::Block { .. } if finished.is_none() => {
                finished = component_row_bytes(domain, indexer)?;
            }
            _ => {}
        }
    }
    let (Some(raw), Some(finished)) = (raw, finished) else {
        return Err(demand_error(format!(
            "selected attention layer {layer} has no indexer history"
        )));
    };
    let block = shape.block;
    let local = raw
        .checked_mul(block)
        .and_then(|bytes| bytes.checked_add(finished))
        .ok_or_else(|| demand_error("indexer row bytes overflow"))?;
    step.launch(local)?;
    let scored = finished
        .checked_add(2 * 4)
        .ok_or_else(|| demand_error("indexer block bytes overflow"))?;
    step.read(scored.div_ceil(block), None)
}

/// The bytes of `count` recurrent bank components from `first`.
fn bank_bytes(
    layout: &ModelStateLayout,
    first: usize,
    count: usize,
    index: usize,
) -> Result<u64, AssessmentError> {
    layout
        .target_recurrent
        .get(first..first + count)
        .ok_or_else(|| demand_error(format!("block {index} has no recurrent state")))?
        .iter()
        .try_fold(0u64, |total, component| {
            total
                .checked_add(usize_bytes(component.bytes().map_err(demand_error)?)?)
                .ok_or_else(|| demand_error("recurrent state bytes overflow"))
        })
}

impl DecodeDemand {
    /// What one plain decode step of one row streams and launches, from the
    /// model's program plan, resident load plan and state layout.
    pub fn from_model(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
    ) -> Result<Self, AssessmentError> {
        let program = load
            .program_plan(definition, codec)
            .map_err(|error| demand_error(error.to_string()))?;
        let layout =
            ModelStateLayout::derive(&definition.decoder, &[], codec, 0).map_err(demand_error)?;
        let target = program.target();
        let mut step = Step::default();

        let embedding = target.embedding();
        let table = planned(
            load,
            WeightScope::Target,
            WeightKind::Embedding,
            embedding.table,
        )?;
        step.launch(table.resident_bytes / table.shape[0])?;
        if let Some(entry) = target.per_layer() {
            per_layer_entry(&mut step, load, entry)?;
        }

        // The first bank component of the next recurrent mixer.
        let mut recurrent_component = 0usize;
        for (index, block) in target.blocks().iter().enumerate() {
            let layer =
                u32::try_from(index).map_err(|_| demand_error("block index exceeds u32"))?;
            let [mixer_scope, feed_forward_scope] =
                crate::programs::native_target_graph::block_scopes(index).map_err(demand_error)?;
            let streams = block.streams();
            // Block 0 widens the embedded row into the streams
            // (`repeat_rows`), which streams no weight.
            if index == 0 && streams.is_some() {
                step.launch(0)?;
            }
            let input = |step: &mut Step, scope, norm| {
                sublayer_input(step, load, definition, scope, norm, streams)
            };
            let scope = mixer_scope;
            let weight = |kind, bound| planned(load, scope, kind, bound);
            match block.mixer() {
                MixerProgramSlot::Attention(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    let shape = binding.shape;
                    let query = if shape.interleaved_gate > 0 {
                        WeightKind::QueryGate
                    } else {
                        WeightKind::Query
                    };
                    let projections = [
                        (true, query, binding.query),
                        (
                            shape.gate_rows() > 0,
                            WeightKind::AttentionGate,
                            binding.gate,
                        ),
                        (shape.key_rows() > 0, WeightKind::Key, binding.key),
                        (shape.value_rows() > 0, WeightKind::Value, binding.value),
                    ]
                    .into_iter()
                    .filter(|(present, _, _)| *present)
                    .map(|(_, kind, bound)| weight(kind, bound))
                    .collect::<Result<Vec<_>, _>>()?;
                    step.project(&projections)?;
                    if binding.history.encoding() == magnitude_state::HistoryEncoding::Unsupported {
                        return Err(demand_error(
                            "rotated K4/V4 history has no native attention entry",
                        ));
                    }
                    let (row_bytes, window) =
                        history_read(&layout.target_history, LayerRef::Target(layer))?;
                    match binding.selection {
                        None => step.attend(row_bytes, window)?,
                        // The indexer projects its query and key, selects,
                        // and attention reads the selected blocks and the
                        // row's tail: at most one block past them.
                        Some(selection) => {
                            step.project(&[
                                weight(WeightKind::IndexerQuery, selection.query)?,
                                weight(WeightKind::IndexerKey, selection.key)?,
                            ])?;
                            indexer_select(
                                &mut step,
                                &layout.target_history,
                                layer,
                                selection.shape,
                            )?;
                            let read = (selection.shape.blocks + 1)
                                .checked_mul(selection.shape.block)
                                .ok_or_else(|| demand_error("selected tokens overflow"))?;
                            step.attend(
                                row_bytes,
                                Some(window.map_or(read, |window| window.min(read))),
                            )?;
                        }
                    }
                    let output = weight(WeightKind::AttentionOutput, binding.output)?;
                    tail(&mut step, binding.tail, output, weight)?;
                }
                // The query-only projection row, the state advance over the
                // layer's bank (window, state and tape components), the gated
                // group norm and the output projection
                // (`operators::state_space`).
                MixerProgramSlot::StateSpace(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    step.project(&[weight(
                        WeightKind::StateSpaceProjection,
                        binding.projection,
                    )?])?;
                    step.launch(bank_bytes(&layout, recurrent_component, 3, index)?)?;
                    recurrent_component += 3;
                    step.launch(
                        weight(WeightKind::StateSpaceNorm, Element::f32())?.resident_bytes,
                    )?;
                    step.project(&[weight(WeightKind::RecurrentOutput, binding.output)?])?;
                }
                // The segmented `u | C` projection, the gated taps over the
                // layer's window (its only bank component) and the output
                // projection (`operators::short_conv`).
                MixerProgramSlot::ShortConv(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    step.project(&[
                        weight(WeightKind::ShortConvInputGate, binding.input_gate)?,
                        weight(WeightKind::ShortConvOutputGate, binding.output_gate)?,
                        weight(WeightKind::ShortConvValue, binding.value)?,
                    ])?;
                    step.launch(bank_bytes(&layout, recurrent_component, 1, index)?)?;
                    recurrent_component += 1;
                    weight(WeightKind::RecurrentConvolution, Element::f32())?;
                    step.project(&[weight(WeightKind::RecurrentOutput, binding.output)?])?;
                }
                // The segmented projection, the step over the layer's
                // recurrent bank (window, delta and tape components) and the
                // output projection.
                MixerProgramSlot::Recurrent(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    step.project(&[
                        weight(WeightKind::RecurrentQueryKeyValue, binding.qkv)?,
                        weight(WeightKind::RecurrentGate, binding.gate)?,
                        weight(WeightKind::RecurrentAlpha, binding.alpha)?,
                        weight(WeightKind::RecurrentBeta, binding.beta)?,
                    ])?;
                    step.launch(bank_bytes(&layout, recurrent_component, 3, index)?)?;
                    recurrent_component += 3;
                    weight(WeightKind::RecurrentNorm, binding.recurrent_norm)?;
                    step.project(&[weight(WeightKind::RecurrentOutput, binding.output)?])?;
                }
            }
            let scope = feed_forward_scope;
            let weight = |kind, bound| planned(load, scope, kind, bound);
            let Some(feed_forward) = block.feed_forward() else {
                continue;
            };
            match feed_forward {
                FeedForwardProgramSlot::Dense(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    step.project(&[
                        weight(WeightKind::DenseGate, binding.gate)?,
                        weight(WeightKind::DenseUp, binding.up)?,
                    ])?;
                    let down = weight(WeightKind::DenseDown, binding.down)?;
                    tail(&mut step, binding.tail, down, weight)?;
                }
                FeedForwardProgramSlot::Routed(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    step.launch(weight(WeightKind::Router, binding.router)?.resident_bytes)?;
                    let chosen = |weight: &WeightPlan| {
                        selected_bytes(weight, binding.experts, binding.selected)
                    };
                    let shared_expansion = [
                        weight(WeightKind::SharedGate, binding.shared_gate)?.resident_bytes,
                        weight(WeightKind::SharedUp, binding.shared_up)?.resident_bytes,
                    ];
                    let expansion = [
                        chosen(weight(WeightKind::ExpertGate, binding.expert_gate)?)?,
                        chosen(weight(WeightKind::ExpertUp, binding.expert_up)?)?,
                    ]
                    .into_iter()
                    .chain(shared_expansion)
                    .try_fold(0u64, u64::checked_add)
                    .ok_or_else(|| demand_error("routed expansion bytes overflow"))?;
                    step.launch(expansion)?;
                    let output = chosen(weight(WeightKind::ExpertDown, binding.expert_down)?)?
                        .checked_add(
                            weight(WeightKind::SharedDown, binding.shared_down)?.resident_bytes,
                        )
                        .ok_or_else(|| demand_error("routed output bytes overflow"))?;
                    step.launch(output)?;
                }
                // `programs::graph::parallel`: the dense branch's expansion
                // and its projection into F32 rows, the routed branch (summed
                // onto zeros) in its own scope, then `moe_tail`.
                FeedForwardProgramSlot::Parallel(binding) => {
                    let WeightScope::TargetSublayer(sublayer) = scope else {
                        return Err(demand_error("parallel branches outside a target sublayer"));
                    };
                    let [dense_scope, routed_scope] = DenseBesideRouted::scopes(sublayer);
                    let dense = binding.dense;
                    let weight = |kind, bound| planned(load, dense_scope, kind, bound);
                    weight(WeightKind::InputNorm, dense.norm)?;
                    step.project(&[
                        weight(WeightKind::DenseGate, dense.gate)?,
                        weight(WeightKind::DenseUp, dense.up)?,
                    ])?;
                    step.project(&[weight(WeightKind::DenseDown, dense.down)?])?;
                    planned(
                        load,
                        routed_scope,
                        WeightKind::InputNorm,
                        binding.routed.norm,
                    )?;
                    general_routed(&mut step, load, routed_scope, &binding.routed)?;
                    // The branch and tail norms share the tail's element.
                    let mut norms = 0u64;
                    for norm_scope in [dense_scope, routed_scope, scope] {
                        norms = norms
                            .checked_add(
                                planned(load, norm_scope, WeightKind::PostNorm, binding.norm)?
                                    .resident_bytes,
                            )
                            .ok_or_else(|| demand_error("tail norm bytes overflow"))?;
                    }
                    step.launch(norms)?;
                }
                FeedForwardProgramSlot::GeneralRouted(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    general_routed(&mut step, load, scope, &binding)?;
                }
            }
            // `programs::graph::per_layer`: the gate over the layer's slice
            // of the per-layer rows, then its post-norm tail.
            if let Some(binding) = block.per_layer() {
                let scope = WeightScope::TargetSublayer(SublayerIndex {
                    block: layer,
                    sublayer: 2,
                });
                let SublayerTail::PostNorm { norm, .. } = binding.tail else {
                    return Err(demand_error(
                        "a per-layer input sublayer without a post-norm tail",
                    ));
                };
                step.project(&[planned(
                    load,
                    scope,
                    WeightKind::PerLayerGate,
                    binding.gate,
                )?])?;
                post_norm(
                    &mut step,
                    planned(
                        load,
                        scope,
                        WeightKind::PerLayerProjection,
                        binding.projection,
                    )?,
                    planned(load, scope, WeightKind::PostNorm, norm)?,
                )?;
            }
            if let Some(binding) = block.ngram() {
                let scope = WeightScope::TargetSublayer(SublayerIndex {
                    block: layer,
                    sublayer: 2,
                });
                ngram(
                    &mut step,
                    load,
                    &layout,
                    scope,
                    binding,
                    recurrent_component,
                    index,
                )?;
                recurrent_component += 2;
            }
        }

        // The last block's graph ends with the decoder's exit site, which
        // injects into nothing.
        if let Some(streams) = target.blocks().last().and_then(|block| block.streams()) {
            mix_site(&mut step, load, WeightScope::Target, streams, false)?;
        }

        // The selection readout graph: the features over the final row under
        // the exit norm (or the exit site's partials, unnormalized), the
        // vocabulary projection (its physical matrix, even when tied to the
        // embedding), and sampling of the one F32 logits row.
        let readout = target.readout();
        let norm = if InputForm::exit(&definition.decoder).normalized {
            planned(
                load,
                WeightScope::Target,
                WeightKind::OutputNorm,
                readout.norm,
            )?
            .resident_bytes
        } else {
            0
        };
        step.launch(norm)?;
        match readout.head {
            // Admission keeps the canonical full-projection demand even
            // when bounded classes have an optional qualified implementation.
            ReadoutHead::Packed { weight, .. } | ReadoutHead::Qualified { weight, .. } => {
                step.project(&[planned(
                    load,
                    WeightScope::Target,
                    WeightKind::Output,
                    weight,
                )?])?;
            }
        }
        step.launch(
            definition
                .decoder
                .vocabulary
                .checked_mul(4)
                .ok_or_else(|| demand_error("logits bytes overflow"))?,
        )?;
        Ok(Self {
            streamed_bytes: step.streamed_bytes,
            history: step.history,
            launches: step.launches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::Layout;

    fn fixture() -> (ModelDefinition, ModelLoadPlan) {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        (definition, load)
    }

    fn resident(load: &ModelLoadPlan, kind: WeightKind) -> &WeightPlan {
        load.target()
            .iter()
            .find(|weight| weight.role.kind == kind)
            .unwrap()
    }

    #[test]
    fn a_plain_decode_step_streams_every_planned_weight_once() {
        let (definition, load) = fixture();
        let demand = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        // Embedding, attention projection, attention, attention output,
        // dense expansion, dense output, readout features, readout head,
        // sampling.
        assert_eq!(demand.launches, 9);
        let weights = [
            WeightKind::QueryGate,
            WeightKind::Key,
            WeightKind::Value,
            WeightKind::AttentionOutput,
            WeightKind::DenseGate,
            WeightKind::DenseUp,
            WeightKind::DenseDown,
            WeightKind::OutputNorm,
            WeightKind::Output,
        ]
        .iter()
        .map(|kind| resident(&load, *kind).resident_bytes)
        .sum::<u64>();
        let vocabulary = definition.decoder.vocabulary;
        let embedding = resident(&load, WeightKind::Embedding);
        let embedding_row = embedding.resident_bytes / embedding.shape[0];
        assert_eq!(
            demand.streamed_bytes,
            weights + embedding_row + vocabulary * 4
        );
    }

    /// Over hyper-connection streams a sublayer reads through its mix site
    /// and the exit site precedes the readout, which reads no norm.
    #[test]
    fn a_streams_decode_step_streams_its_mix_sites_and_no_norms() {
        let definition = crate::planning::tests::hyper_connection_fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let demand = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        // Embedding, widening, mix site (two stages), attention projection,
        // attention, attention output, exit site (two stages), readout
        // features, readout head, sampling.
        assert_eq!(demand.launches, 12);
        let mixer = WeightScope::TargetSublayer(SublayerIndex {
            block: 0,
            sublayer: 0,
        });
        let bytes = |scope, kind| {
            load.target()
                .iter()
                .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
                .unwrap()
                .resident_bytes
        };
        // Each site streams its norm with both stages.
        let site = |scope, injected: bool| {
            2 * bytes(scope, WeightKind::HyperConnectionNorm)
                + bytes(scope, WeightKind::HyperConnectionDown)
                + bytes(scope, WeightKind::HyperConnectionUp)
                + if injected {
                    bytes(scope, WeightKind::HyperConnectionInject)
                } else {
                    0
                }
        };
        let weights = [
            WeightKind::QueryGate,
            WeightKind::Key,
            WeightKind::Value,
            WeightKind::AttentionOutput,
        ]
        .iter()
        .map(|kind| bytes(mixer, *kind))
        .sum::<u64>()
            + bytes(WeightScope::Target, WeightKind::Output);
        let embedding = resident(&load, WeightKind::Embedding);
        let embedding_row = embedding.resident_bytes / embedding.shape[0];
        assert_eq!(
            demand.streamed_bytes,
            weights
                + site(mixer, true)
                + site(WeightScope::Target, false)
                + embedding_row
                + definition.decoder.vocabulary * 4
        );
    }

    /// An n-gram input charges its row rule over its token window, its
    /// heads' stored rows, its key and value projections, its gate's norms,
    /// and its row op's norm, taps and window; the recurrent mixers after it
    /// read their own bank components.
    #[test]
    fn an_ngram_input_streams_its_rows_projections_and_windows() {
        let load_of = |definition: &ModelDefinition| {
            let manifest = crate::planning::tests::stored_table_manifest(definition);
            ModelLoadPlan::derive(
                &manifest,
                definition,
                crate::ComponentSelection {
                    head: false,
                    vision: false,
                },
                Layout::Rows16,
            )
            .unwrap()
        };
        let definition = crate::planning::tests::qwen4exp_fixture_definition();
        let mut plain = definition.clone();
        plain.decoder.blocks[0].sublayers.truncate(2);
        let load = load_of(&definition);
        let with = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let without = DecodeDemand::from_model(&plain, &load_of(&plain), KvCodec::Dense).unwrap();
        // Embedding and widening; per block a mix site (two stages) before
        // its mixer and its feed-forward; the gated delta's projection, step
        // and output; the routed router, expansion and output; the
        // attention's projection, read and output; the exit site and the
        // readout's three.
        assert_eq!(without.launches, 2 + 2 * 2 * 2 + 3 + 3 + 3 + 3 + 2 + 3);
        assert_eq!(with.launches, without.launches + 6);
        let scope = WeightScope::TargetSublayer(SublayerIndex {
            block: 0,
            sublayer: 2,
        });
        let bytes = |kind| {
            load.target()
                .iter()
                .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
                .unwrap()
        };
        let table = bytes(WeightKind::NGramTable);
        let heads = 4;
        let layout = ModelStateLayout::derive(&definition.decoder, &[], KvCodec::Dense, 0).unwrap();
        // The n-gram's windows follow the gated delta's three components.
        let windows = layout.target_recurrent[3..5]
            .iter()
            .map(|component| component.bytes().unwrap() as u64)
            .sum::<u64>();
        let ngram = heads * table.resident_bytes / table.shape[0]
            + [
                WeightKind::NGramKey,
                WeightKind::NGramValue,
                WeightKind::NGramKeyNorm,
                WeightKind::NGramQueryNorm,
                WeightKind::NGramConvolutionNorm,
                WeightKind::NGramConvolution,
            ]
            .iter()
            .map(|kind| bytes(*kind).resident_bytes)
            .sum::<u64>()
            + windows;
        assert_eq!(with.streamed_bytes, without.streamed_bytes + ngram);
    }

    #[test]
    fn history_bytes_per_token_follow_the_state_layout_and_codec() {
        let (definition, load) = fixture();
        // One kv head of width 64: dense bf16 keys and values, 2 × 64 × 2.
        let dense = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        assert_eq!(
            dense.history,
            [HistoryRead {
                bytes_per_token: 256,
                window: None,
            }]
        );
        // Affine: 64 key bytes, 32 value bytes and two (scale, zero) f16
        // pairs per 32-value group for each.
        let affine = DecodeDemand::from_model(&definition, &load, KvCodec::AffineK8V4).unwrap();
        assert_eq!(
            affine.history,
            [HistoryRead {
                bytes_per_token: 64 + 32 + 2 * (2 * 2 * 2),
                window: None,
            }]
        );
        assert!(matches!(
            DecodeDemand::from_model(&definition, &load, KvCodec::RotatedK4V4),
            Err(AssessmentError::Demand(_))
        ));
    }

    /// A selected read charges its indexer: the projection of its query and
    /// key, a select call writing its raw key and block and reading its
    /// block's raw keys, and per token one block's F32 key and score traffic
    /// over the block's rows; attention reads the selected blocks and the
    /// tail alone.
    #[test]
    fn a_selected_read_charges_its_indexer_and_reads_its_blocks() {
        use magnitude_family_contracts::{
            HistoryReads, Indexer, Operator, RmsNorm, WeightDescriptor,
        };
        let (mut definition, _) = fixture();
        let hidden = definition.decoder.hidden;
        let Operator::Attention(attention) = &mut definition.decoder.blocks[0].sublayers[0].op
        else {
            panic!("the fixture's mixer is attention");
        };
        let norm = |name: &str| RmsNorm {
            weight: WeightDescriptor::stored(name, vec![64]),
            epsilon: 1e-6,
        };
        attention.reads = HistoryReads::Selected {
            block: 4,
            blocks: 8,
            indexer: Indexer {
                heads: 2,
                width: 64,
                query: WeightDescriptor::stored("indexer_q", vec![128, hidden]),
                key: WeightDescriptor::stored("indexer_k", vec![64, hidden]),
                query_norm: norm("indexer_q_norm"),
                key_norm: norm("indexer_k_norm"),
            },
        };
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let (plain_definition, plain_load) = fixture();
        let plain =
            DecodeDemand::from_model(&plain_definition, &plain_load, KvCodec::Dense).unwrap();
        let selected = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        // The indexer's projection and select calls.
        assert_eq!(selected.launches, plain.launches + 2);
        // Its projection weights; the select call's raw keys (four F32 rows
        // of 64 plus their coordinates) and finished block key.
        let projection = resident(&load, WeightKind::IndexerQuery).resident_bytes
            + resident(&load, WeightKind::IndexerKey).resident_bytes;
        let raw = 64 * 4 + 4 * 4;
        assert_eq!(
            selected.streamed_bytes,
            plain.streamed_bytes + projection + 4 * raw + 64 * 4
        );
        // Attention reads at most the eight selected blocks and its tail;
        // the indexer one block key and two scores per four tokens.
        assert_eq!(
            selected.history,
            [
                HistoryRead {
                    bytes_per_token: (64 * 4 + 8) / 4,
                    window: None,
                },
                HistoryRead {
                    bytes_per_token: 256,
                    window: Some(9 * 4),
                },
            ]
        );
    }

    #[test]
    fn a_window_reads_at_most_its_tokens() {
        let read = HistoryRead {
            bytes_per_token: 10,
            window: Some(100),
        };
        assert_eq!(read.bytes_at(40), 400);
        assert_eq!(read.bytes_at(4_000), 1_000);
    }

    #[test]
    fn a_selected_expert_share_streams_its_slabs() {
        let (_, load) = fixture();
        let mut experts = resident(&load, WeightKind::DenseUp).clone();
        experts.shape = vec![8, 32, 64];
        experts.resident_bytes = 8 * 32 * 64 * 2;
        assert_eq!(selected_bytes(&experts, 8, 2).unwrap(), 2 * 32 * 64 * 2);
        assert!(selected_bytes(&experts, 3, 1).is_err());
    }
}
