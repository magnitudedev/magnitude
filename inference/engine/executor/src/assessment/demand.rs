//! Header-derived decode demand: the bytes one plain target decode step
//! streams, the history it reads per token of context, and the entry calls
//! it launches. Derived from the same program and load plans native
//! preparation uses, and from the state layout the memory terms use for
//! history bytes.

use super::AssessmentError;
use crate::operators::parallel::DenseBesideRouted;
use crate::{
    FeedForwardProgramSlot, GeneralRoutedBinding, MixerProgramSlot, ModelLoadPlan,
    PerLayerEntryBinding, SublayerTail, WeightPlan,
};
use magnitude_family_contracts::{ModelDefinition, SublayerIndex, WeightKind, WeightScope};
use magnitude_state::{HistoryDomainLayout, KvCodec, LayerRef, ModelStateLayout};
use seismic::Element;

/// History one decode step reads for each token of context, summed over the
/// attention layers that share a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryRead {
    pub bytes_per_token: u64,
    /// A window domain reads at most its window's tokens. `None` reads every
    /// token.
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
fn selected_bytes(weight: &WeightPlan, experts: u64, selected: u64) -> Result<u64, AssessmentError> {
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
            .try_fold(0u64, |total, weight| total.checked_add(weight.resident_bytes))
            .ok_or_else(|| demand_error("projection bytes overflow"))?;
        self.launch(bytes)
    }

    /// One attention entry call reading `bytes_per_token` of history per
    /// token of context, up to `window` tokens.
    fn attend(&mut self, bytes_per_token: u64, window: Option<u64>) -> Result<(), AssessmentError> {
        self.launch(0)?;
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
fn post_norm(step: &mut Step, output: &WeightPlan, norm: &WeightPlan) -> Result<(), AssessmentError> {
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

/// `operators::routed` in `scope`: the shared expert onto the sum's root,
/// selection, the latent projections, and the selected experts' expansion
/// and down projection.
fn general_routed(
    step: &mut Step,
    load: &ModelLoadPlan,
    scope: WeightScope,
    binding: &GeneralRoutedBinding,
) -> Result<(), AssessmentError> {
    let weight = |kind, bound| planned(load, scope, kind, bound);
    let shape = binding.shape;
    weight(WeightKind::InputNorm, binding.norm)?;
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
    step.launch(chosen(weight(WeightKind::ExpertDown, binding.expert_down)?)?)
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
            HistoryDomainLayout::Block { .. } => {
                return Err(demand_error("block history domains have no decode demand"))
            }
            HistoryDomainLayout::Shared { source, layers } => {
                if layers.contains(&layer) {
                    return history_read(domains, *source);
                }
                continue;
            }
        };
        if let Some(component) = domain
            .components()
            .iter()
            .find(|component| component.layer == layer)
        {
            let row_bytes = component.planes().iter().try_fold(0u64, |total, plane| {
                total
                    .checked_add(usize_bytes(plane.row_bytes)?)
                    .ok_or_else(|| demand_error("history row bytes overflow"))
            })?;
            return Ok((row_bytes, window));
        }
    }
    Err(demand_error(format!("attention layer {layer:?} has no history")))
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
        let vocabulary = definition.decoder.vocabulary;
        step.launch(table.resident_bytes / vocabulary)?;
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
            let scope = mixer_scope;
            let weight = |kind, bound| planned(load, scope, kind, bound);
            match block.mixer() {
                MixerProgramSlot::Attention(binding) => {
                    weight(WeightKind::InputNorm, binding.norm)?;
                    let shape = binding.shape;
                    let query = if shape.interleaved_gate > 0 {
                        WeightKind::QueryGate
                    } else {
                        WeightKind::Query
                    };
                    let projections = [
                        (true, query, binding.query),
                        (shape.gate_rows() > 0, WeightKind::AttentionGate, binding.gate),
                        (shape.key_rows() > 0, WeightKind::Key, binding.key),
                        (shape.value_rows() > 0, WeightKind::Value, binding.value),
                    ]
                    .into_iter()
                    .filter(|(present, _, _)| *present)
                    .map(|(_, kind, bound)| weight(kind, bound))
                    .collect::<Result<Vec<_>, _>>()?;
                    step.project(&projections)?;
                    if binding.history == KvCodec::RotatedK4V4 {
                        return Err(demand_error(
                            "rotated K4/V4 history has no native attention entry",
                        ));
                    }
                    let (row_bytes, window) =
                        history_read(&layout.target_history, LayerRef::Target(layer))?;
                    step.attend(row_bytes, window)?;
                    let output = weight(WeightKind::AttentionOutput, binding.output)?;
                    tail(&mut step, binding.tail, output, weight)?;
                }
                // The query-only projection row, the state advance over the
                // layer's bank (window, state and tape components), the gated
                // group norm and the output projection
                // (`operators::state_space`).
                MixerProgramSlot::StateSpace(binding) => {
                    weight(WeightKind::InputNorm, binding.norm)?;
                    step.project(&[weight(WeightKind::StateSpaceProjection, binding.projection)?])?;
                    step.launch(bank_bytes(&layout, recurrent_component, 3, index)?)?;
                    recurrent_component += 3;
                    step.launch(weight(WeightKind::StateSpaceNorm, Element::f32())?.resident_bytes)?;
                    step.project(&[weight(WeightKind::RecurrentOutput, binding.output)?])?;
                }
                // The segmented `u | C` projection, the gated taps over the
                // layer's window (its only bank component) and the output
                // projection (`operators::short_conv`).
                MixerProgramSlot::ShortConv(binding) => {
                    weight(WeightKind::InputNorm, binding.norm)?;
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
                    weight(WeightKind::InputNorm, binding.norm)?;
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
                    weight(WeightKind::InputNorm, binding.norm)?;
                    step.project(&[
                        weight(WeightKind::DenseGate, binding.gate)?,
                        weight(WeightKind::DenseUp, binding.up)?,
                    ])?;
                    let down = weight(WeightKind::DenseDown, binding.down)?;
                    tail(&mut step, binding.tail, down, weight)?;
                }
                FeedForwardProgramSlot::Routed(binding) => {
                    weight(WeightKind::InputNorm, binding.norm)?;
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
                    return Err(demand_error("a per-layer input sublayer without a post-norm tail"));
                };
                step.project(&[planned(load, scope, WeightKind::PerLayerGate, binding.gate)?])?;
                post_norm(
                    &mut step,
                    planned(load, scope, WeightKind::PerLayerProjection, binding.projection)?,
                    planned(load, scope, WeightKind::PostNorm, norm)?,
                )?;
            }
        }

        // The selection readout graph: final-norm features, the vocabulary
        // projection (its physical matrix, even when tied to the embedding),
        // and sampling of the one F32 logits row.
        let readout = target.readout();
        step.launch(
            planned(
                load,
                WeightScope::Target,
                WeightKind::OutputNorm,
                readout.norm,
            )?
            .resident_bytes,
        )?;
        step.project(&[planned(
            load,
            WeightScope::Target,
            WeightKind::Output,
            readout.weight,
        )?])?;
        step.launch(
            vocabulary
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
        let embedding_row = resident(&load, WeightKind::Embedding).resident_bytes / vocabulary;
        assert_eq!(
            demand.streamed_bytes,
            weights + embedding_row + vocabulary * 4
        );
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
