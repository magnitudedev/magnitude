//! Header-derived decode demand: every launch of one plain target decode
//! step, keyed by the measured class it costs against, with how often it
//! launches and how many bytes it streams. Derived from the same program and
//! load plans native preparation uses, and from the state layout the memory
//! terms use for history bytes.

use super::basis::{ClassMeasurement, HeadGeometry, MeasurementBasis, MeasurementKey};
use super::AssessmentError;
use crate::operators::parallel::DenseBesideRouted;
use crate::{
    FeedForwardProgramSlot, GeneralRoutedBinding, MixerProgramSlot, ModelLoadPlan,
    PerLayerEntryBinding, SublayerTail, WeightPlan,
};
use magnitude_family_contracts::{ModelDefinition, SublayerIndex, WeightKind, WeightScope};
use magnitude_state::{HistoryDomainLayout, KvCodec, LayerRef, ModelStateLayout};
use seismic::Element;

/// The geometry a term's class cost is read at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermShape {
    /// A class whose cost follows its bytes and launches.
    Plain,
    /// One weight segment of weight-streaming launches producing
    /// `launch_rows` output rows each (over every segment of the launch),
    /// streamed in `weight`.
    Projection { weight: Element, launch_rows: u64 },
    /// Decode attention at its head geometry.
    Attention(HeadGeometry),
}

/// One measured class's share of a plain decode step. Streamed bytes grow
/// with context depth only for history-reading classes: linearly, up to a
/// window domain's window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemandTerm {
    /// The entry and its exact bindings.
    pub key: MeasurementKey,
    pub launches: u64,
    /// Bytes streamed per step independent of depth, summed over launches.
    pub bytes: u64,
    /// Additional bytes streamed per step for each token of context.
    pub bytes_per_context_token: u64,
    /// The most context tokens those bytes are streamed for: a window
    /// domain's history reads at most its window. `None` reads every token.
    pub context_window: Option<u64>,
    pub shape: TermShape,
}

impl DemandTerm {
    /// The representation a weight-streaming term streams.
    pub fn weight(&self) -> Option<Element> {
        match self.shape {
            TermShape::Projection { weight, .. } => Some(weight),
            TermShape::Plain | TermShape::Attention(_) => None,
        }
    }
}

/// Every launch of one plain target decode step, grouped by measured class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeDemand {
    pub terms: Vec<DemandTerm>,
}

fn demand_error(message: impl Into<String>) -> AssessmentError {
    AssessmentError::Demand(message.into())
}

/// One segment of a weight-streaming launch: the weight streamed, the output
/// rows it produces and the bytes it streams.
struct Segment<'a> {
    weight: &'a WeightPlan,
    rows: u64,
    bytes: u64,
}

impl<'a> Segment<'a> {
    /// The whole weight: its output rows are every leading row of the
    /// matrix.
    fn whole(weight: &'a WeightPlan) -> Result<Self, AssessmentError> {
        Ok(Self {
            weight,
            rows: output_rows(weight)?,
            bytes: weight.resident_bytes,
        })
    }

    /// `selected` of an expert tensor's `experts` equal slabs: a decode row
    /// streams only its selected experts.
    fn selected(
        weight: &'a WeightPlan,
        experts: u64,
        selected: u64,
    ) -> Result<Self, AssessmentError> {
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
        let overflow = || demand_error("selected expert streaming overflows");
        Ok(Self {
            weight,
            rows: (rows / experts).checked_mul(selected).ok_or_else(overflow)?,
            bytes: (weight.resident_bytes / experts)
                .checked_mul(selected)
                .ok_or_else(overflow)?,
        })
    }
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

/// Terms in first-launch order; terms of equal keys, shapes and windows
/// accumulate.
#[derive(Default)]
struct Terms(Vec<DemandTerm>);

impl Terms {
    /// Launches of a class whose cost follows its bytes.
    fn add(&mut self, key: MeasurementKey, launches: u64, bytes: u64) -> Result<(), AssessmentError> {
        self.add_term(DemandTerm {
            key,
            launches,
            bytes,
            bytes_per_context_token: 0,
            context_window: None,
            shape: TermShape::Plain,
        })
    }

    /// One history-reading launch: `row_bytes` per token of context, up to
    /// `window` tokens when the history is a window domain's.
    fn add_history(
        &mut self,
        key: MeasurementKey,
        heads: HeadGeometry,
        row_bytes: u64,
        window: Option<u64>,
    ) -> Result<(), AssessmentError> {
        self.add_term(DemandTerm {
            key,
            launches: 1,
            bytes: 0,
            bytes_per_context_token: row_bytes,
            context_window: window,
            shape: TermShape::Attention(heads),
        })
    }

    /// Accumulate `new` into the term of equal key, shape and window.
    fn add_term(&mut self, new: DemandTerm) -> Result<(), AssessmentError> {
        let overflow = || demand_error(format!("{} demand overflows", new.key.class.name()));
        match self.0.iter_mut().find(|term| {
            term.key == new.key
                && term.shape == new.shape
                && term.context_window == new.context_window
        }) {
            Some(term) => {
                term.launches = term.launches.checked_add(new.launches).ok_or_else(overflow)?;
                term.bytes = term.bytes.checked_add(new.bytes).ok_or_else(overflow)?;
                term.bytes_per_context_token = term
                    .bytes_per_context_token
                    .checked_add(new.bytes_per_context_token)
                    .ok_or_else(overflow)?;
            }
            None => self.0.push(new),
        }
        Ok(())
    }

    /// One launch of a weight-streaming entry of one or more segments: each
    /// segment streams its representation at the whole launch's output rows,
    /// and the launch is counted once, on the first segment.
    fn project(
        &mut self,
        segments: &[Segment<'_>],
        key: impl Fn(Element) -> MeasurementKey,
    ) -> Result<(), AssessmentError> {
        let launch_rows = segments
            .iter()
            .try_fold(0u64, |total, segment| total.checked_add(segment.rows))
            .ok_or_else(|| demand_error("launch rows overflow"))?;
        for (index, segment) in segments.iter().enumerate() {
            let weight = segment.weight.resident;
            self.add_term(DemandTerm {
                key: key(weight),
                launches: u64::from(index == 0),
                bytes: segment.bytes,
                bytes_per_context_token: 0,
                context_window: None,
                shape: TermShape::Projection {
                    weight,
                    launch_rows,
                },
            })?;
        }
        Ok(())
    }

    /// One launch of a weight-streaming entry over one whole weight.
    fn project_whole(
        &mut self,
        weight: &WeightPlan,
        key: impl Fn(Element) -> MeasurementKey,
    ) -> Result<(), AssessmentError> {
        self.project(&[Segment::whole(weight)?], key)
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

/// A sublayer's output terms by its tail: the operator's own output entry
/// (`residual`), or the `output` projection into F32 rows and the post-norm
/// row op, whose norm weight `planned` checks.
fn tail<'a>(
    terms: &mut Terms,
    tail: SublayerTail,
    residual: impl Fn(Element) -> MeasurementKey,
    output: &WeightPlan,
    activation: Element,
    planned: impl Fn(WeightKind, Element) -> Result<&'a WeightPlan, AssessmentError>,
) -> Result<(), AssessmentError> {
    match tail {
        SublayerTail::Residual => terms.project_whole(output, residual),
        SublayerTail::PostNorm { norm, .. } => {
            post_norm(terms, output, planned(WeightKind::PostNorm, norm)?, activation)
        }
    }
}

/// A post-norm tail: the `output` projection into F32 rows, then the row op
/// normalizing it with `norm` into the residual.
fn post_norm(
    terms: &mut Terms,
    output: &WeightPlan,
    norm: &WeightPlan,
    activation: Element,
) -> Result<(), AssessmentError> {
    terms.project_whole(output, |weight| {
        MeasurementKey::project_rows(weight, activation)
    })?;
    terms.add(
        MeasurementKey::post_norm_residual(norm.resident),
        1,
        norm.resident_bytes,
    )
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
    terms: &mut Terms,
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
    terms.add(MeasurementKey::table_upload(), 1, table.bytes / rows)?;
    terms.add(
        MeasurementKey::import_rows(Element::f32(), entry.activation),
        1,
        row_bytes(Element::f32(), entry.hidden)?,
    )?;
    let projection = planned(
        load,
        WeightScope::Target,
        WeightKind::PerLayerModelProjection,
        entry.projection,
    )?;
    terms.project_whole(projection, |weight| {
        MeasurementKey::project_rows(weight, entry.activation)
    })?;
    let conversion = if entry.table_source.dtype().is_some() && entry.table.dtype().is_some() {
        MeasurementKey::import_rows(entry.table_source, entry.table)
    } else {
        MeasurementKey::repack_rows(entry.table_source, entry.table)
    };
    terms.add(conversion, 1, row_bytes(entry.table_source, channels)?)?;
    planned(
        load,
        WeightScope::Target,
        WeightKind::PerLayerProjectionNorm,
        entry.norm,
    )?;
    let f32_rows = row_bytes(Element::f32(), channels)?;
    terms.add(
        MeasurementKey::per_layer_inputs(entry.table, entry.norm),
        1,
        f32_rows,
    )?;
    terms.add(MeasurementKey::copy_rows(), 1, f32_rows)
}

/// `operators::routed` in `scope`: the shared expert onto the sum's root,
/// selection, the latent projections, and the selected experts' expansion
/// and down projection.
fn general_routed<'a>(
    terms: &mut Terms,
    load: &'a ModelLoadPlan,
    scope: WeightScope,
    binding: &GeneralRoutedBinding,
) -> Result<(), AssessmentError> {
    let weight = |kind, bound| planned(load, scope, kind, bound);
    let shape = binding.shape;
    let a = binding.activation;
    weight(WeightKind::InputNorm, binding.norm)?;
    if let Some((gate, up, down)) = binding.shared {
        let up = weight(WeightKind::SharedUp, up)?;
        match gate {
            Some(gate) => {
                let gate = weight(WeightKind::SharedGate, gate)?;
                terms.project(&[Segment::whole(gate)?, Segment::whole(up)?], |element| {
                    MeasurementKey::dense_expand(binding.norm, element, a)
                })?;
            }
            None => terms.project_whole(up, |element| {
                MeasurementKey::dense_up(binding.norm, element, a)
            })?,
        }
        let down = weight(WeightKind::SharedDown, down)?;
        terms.project_whole(down, |element| MeasurementKey::dense_output(element, a))?;
    }
    let router = weight(WeightKind::Router, binding.router)?;
    terms.add(
        MeasurementKey::routed_select(binding.norm, binding.router, a),
        1,
        router.resident_bytes,
    )?;
    if let Some((down, up)) = binding.latent {
        let down = weight(WeightKind::LatentDown, down)?;
        terms.project_whole(down, |element| MeasurementKey::project_rows(element, a))?;
        let up = weight(WeightKind::LatentUp, up)?;
        terms.project_whole(up, |element| MeasurementKey::dense_output(element, a))?;
    }
    let chosen =
        |weight: &'a WeightPlan| Segment::selected(weight, shape.experts, shape.selected);
    let expert_up = weight(WeightKind::ExpertUp, binding.expert_up)?;
    let gated = shape.experts_expansion.gated;
    let mut segments = vec![chosen(expert_up)?];
    if let Some(gate) = binding.expert_gate {
        segments.insert(0, chosen(weight(WeightKind::ExpertGate, gate)?)?);
    }
    terms.project(&segments, |element| {
        MeasurementKey::routed_expansion(gated, element, a)
    })?;
    let expert_down = weight(WeightKind::ExpertDown, binding.expert_down)?;
    terms.project(&[chosen(expert_down)?], |element| {
        MeasurementKey::routed_down(element, a)
    })
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

impl DecodeDemand {
    /// Every launch one plain decode step of one row makes, from the model's
    /// program plan, resident load plan and state layout.
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
        let mut terms = Terms::default();

        let embedding = target.embedding();
        let table = planned(
            load,
            WeightScope::Target,
            WeightKind::Embedding,
            embedding.table,
        )?;
        let vocabulary = definition.decoder.vocabulary;
        terms.add(
            MeasurementKey::embedding_rows(embedding.table, embedding.activation),
            1,
            table.resident_bytes / vocabulary,
        )?;
        if let Some(entry) = target.per_layer() {
            per_layer_entry(&mut terms, load, entry)?;
        }

        // The first bank component of the next recurrent mixer.
        let mut recurrent_component = 0usize;
        for (index, block) in target.blocks().iter().enumerate() {
            let layer =
                u32::try_from(index).map_err(|_| demand_error("block index exceeds u32"))?;
            let [mixer_scope, feed_forward_scope] =
                crate::programs::native_target_graph::block_scopes(index).map_err(demand_error)?;
            let scope = mixer_scope;
            match block.mixer() {
                MixerProgramSlot::Attention(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::InputNorm, binding.norm)?;
                    let shape = binding.shape;
                    let query = if shape.interleaved_gate > 0 {
                        WeightKind::QueryGate
                    } else {
                        WeightKind::Query
                    };
                    let segments = [
                        (true, query, binding.query),
                        (shape.gate_rows() > 0, WeightKind::AttentionGate, binding.gate),
                        (shape.key_rows() > 0, WeightKind::Key, binding.key),
                        (shape.value_rows() > 0, WeightKind::Value, binding.value),
                    ]
                    .into_iter()
                    .filter(|(present, _, _)| *present)
                    .map(|(_, kind, bound)| Segment::whole(weight(kind, bound)?))
                    .collect::<Result<Vec<_>, _>>()?;
                    terms.project(&segments, |element| {
                        MeasurementKey::attention_project(binding.norm, element, binding.activation)
                    })?;
                    let affine = match binding.history {
                        KvCodec::Dense => false,
                        KvCodec::AffineK8V4 => true,
                        KvCodec::RotatedK4V4 => {
                            return Err(demand_error(
                                "rotated K4/V4 history has no native attention entry",
                            ));
                        }
                    };
                    let (row_bytes, window) =
                        history_read(&layout.target_history, LayerRef::Target(layer))?;
                    terms.add_history(
                        MeasurementKey::attention_decode(affine, binding.activation),
                        HeadGeometry {
                            kv_heads: shape.kv_heads,
                            group: shape.group,
                            width: shape.width,
                        },
                        row_bytes,
                        window,
                    )?;
                    let output = weight(WeightKind::AttentionOutput, binding.output)?;
                    tail(
                        &mut terms,
                        binding.tail,
                        |element| MeasurementKey::attention_output(element, binding.activation),
                        output,
                        binding.activation,
                        |kind, bound| planned(load, scope, kind, bound),
                    )?;
                }
                // The query-only projection row, the state advance over the
                // layer's bank, the gated group norm and the output
                // projection (`operators::state_space`).
                MixerProgramSlot::StateSpace(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::InputNorm, binding.norm)?;
                    let projection = weight(WeightKind::StateSpaceProjection, binding.projection)?;
                    terms.project_whole(projection, |element| {
                        MeasurementKey::attention_project(binding.norm, element, binding.activation)
                    })?;
                    // The step reads and publishes the layer's bank: window,
                    // state and tape components.
                    let first = recurrent_component;
                    let state = layout
                        .target_recurrent
                        .get(first..first + 3)
                        .ok_or_else(|| {
                            demand_error(format!("state-space block {index} has no state"))
                        })?
                        .iter()
                        .try_fold(0u64, |total, component| {
                            total
                                .checked_add(usize_bytes(component.bytes().map_err(demand_error)?)?)
                                .ok_or_else(|| demand_error("state-space bytes overflow"))
                        })?;
                    recurrent_component += 3;
                    terms.add(
                        MeasurementKey::state_space_step(binding.activation),
                        1,
                        state,
                    )?;
                    let norm = planned(
                        load,
                        scope,
                        WeightKind::StateSpaceNorm,
                        Element::f32(),
                    )?;
                    terms.add(
                        MeasurementKey::state_space_gate(binding.activation),
                        1,
                        norm.resident_bytes,
                    )?;
                    let output = weight(WeightKind::RecurrentOutput, binding.output)?;
                    terms.project_whole(output, |element| {
                        MeasurementKey::attention_output(element, binding.activation)
                    })?;
                }
                // The segmented `u | C` projection, the gated taps over the
                // layer's window and the output projection
                // (`operators::short_conv`).
                MixerProgramSlot::ShortConv(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::InputNorm, binding.norm)?;
                    let segments = [
                        Segment::whole(weight(WeightKind::ShortConvInputGate, binding.input_gate)?)?,
                        Segment::whole(weight(WeightKind::ShortConvOutputGate, binding.output_gate)?)?,
                        Segment::whole(weight(WeightKind::ShortConvValue, binding.value)?)?,
                    ];
                    terms.project(&segments, |element| {
                        MeasurementKey::short_conv_project(binding.norm, element, binding.activation)
                    })?;
                    // The row reads and publishes the layer's window, its
                    // only bank component.
                    let window = layout
                        .target_recurrent
                        .get(recurrent_component)
                        .ok_or_else(|| {
                            demand_error(format!("short convolution block {index} has no window"))
                        })?
                        .bytes()
                        .map_err(demand_error)?;
                    recurrent_component += 1;
                    planned(
                        load,
                        scope,
                        WeightKind::RecurrentConvolution,
                        Element::f32(),
                    )?;
                    terms.add(
                        MeasurementKey::short_conv_rows(binding.activation),
                        1,
                        usize_bytes(window)?,
                    )?;
                    let output = weight(WeightKind::RecurrentOutput, binding.output)?;
                    terms.project_whole(output, |element| {
                        MeasurementKey::attention_output(element, binding.activation)
                    })?;
                }
                MixerProgramSlot::Recurrent(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::InputNorm, binding.norm)?;
                    let segments = [
                        Segment::whole(weight(WeightKind::RecurrentQueryKeyValue, binding.qkv)?)?,
                        Segment::whole(weight(WeightKind::RecurrentGate, binding.gate)?)?,
                        Segment::whole(weight(WeightKind::RecurrentAlpha, binding.alpha)?)?,
                        Segment::whole(weight(WeightKind::RecurrentBeta, binding.beta)?)?,
                    ];
                    terms.project(&segments, |element| {
                        MeasurementKey::delta_project(binding.norm, element, binding.activation)
                    })?;
                    // The step reads and publishes the layer's recurrent bank:
                    // window, delta and tape components.
                    let first = recurrent_component;
                    let state = layout
                        .target_recurrent
                        .get(first..first + 3)
                        .ok_or_else(|| {
                            demand_error(format!("recurrent block {index} has no state"))
                        })?
                        .iter()
                        .try_fold(0u64, |total, component| {
                            total
                                .checked_add(usize_bytes(component.bytes().map_err(demand_error)?)?)
                                .ok_or_else(|| demand_error("recurrent state bytes overflow"))
                        })?;
                    recurrent_component += 3;
                    terms.add(MeasurementKey::delta_step(binding.activation), 1, state)?;
                    planned(
                        load,
                        scope,
                        WeightKind::RecurrentNorm,
                        binding.recurrent_norm,
                    )?;
                    let output = weight(WeightKind::RecurrentOutput, binding.output)?;
                    terms.project_whole(output, |element| {
                        MeasurementKey::delta_output(binding.recurrent_norm, element, binding.activation)
                    })?;
                }
            }
            let scope = feed_forward_scope;
            let Some(feed_forward) = block.feed_forward() else {
                continue;
            };
            match feed_forward {
                FeedForwardProgramSlot::Dense(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::InputNorm, binding.norm)?;
                    let segments = [
                        Segment::whole(weight(WeightKind::DenseGate, binding.gate)?)?,
                        Segment::whole(weight(WeightKind::DenseUp, binding.up)?)?,
                    ];
                    terms.project(&segments, |element| {
                        MeasurementKey::dense_expand(binding.norm, element, binding.activation)
                    })?;
                    let down = weight(WeightKind::DenseDown, binding.down)?;
                    tail(
                        &mut terms,
                        binding.tail,
                        |element| MeasurementKey::dense_output(element, binding.activation),
                        down,
                        binding.activation,
                        |kind, bound| planned(load, scope, kind, bound),
                    )?;
                }
                FeedForwardProgramSlot::Routed(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::InputNorm, binding.norm)?;
                    let router = weight(WeightKind::Router, binding.router)?;
                    terms.add(
                        MeasurementKey::routed_route(
                            binding.norm,
                            binding.router,
                            binding.activation,
                        ),
                        1,
                        router.resident_bytes,
                    )?;
                    let chosen = |weight| Segment::selected(weight, binding.experts, binding.selected);
                    terms.project(
                        &[
                            chosen(weight(WeightKind::ExpertGate, binding.expert_gate)?)?,
                            chosen(weight(WeightKind::ExpertUp, binding.expert_up)?)?,
                            Segment::whole(weight(WeightKind::SharedGate, binding.shared_gate)?)?,
                            Segment::whole(weight(WeightKind::SharedUp, binding.shared_up)?)?,
                        ],
                        |element| MeasurementKey::routed_expand(element, binding.activation),
                    )?;
                    terms.project(
                        &[
                            chosen(weight(WeightKind::ExpertDown, binding.expert_down)?)?,
                            Segment::whole(weight(WeightKind::SharedDown, binding.shared_down)?)?,
                        ],
                        |element| MeasurementKey::routed_output(element, binding.activation),
                    )?;
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
                    let segments = [
                        Segment::whole(weight(WeightKind::DenseGate, dense.gate)?)?,
                        Segment::whole(weight(WeightKind::DenseUp, dense.up)?)?,
                    ];
                    terms.project(&segments, |element| {
                        MeasurementKey::dense_expand(dense.norm, element, dense.activation)
                    })?;
                    let down = weight(WeightKind::DenseDown, dense.down)?;
                    terms.project_whole(down, |element| {
                        MeasurementKey::project_rows(element, dense.activation)
                    })?;
                    general_routed(&mut terms, load, routed_scope, &binding.routed)?;
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
                    terms.add(MeasurementKey::moe_tail(binding.norm), 1, norms)?;
                }
                FeedForwardProgramSlot::GeneralRouted(binding) => {
                    general_routed(&mut terms, load, scope, &binding)?;
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
                let gate = planned(load, scope, WeightKind::PerLayerGate, binding.gate)?;
                terms.project_whole(gate, |element| {
                    MeasurementKey::per_layer_gate(element, binding.activation)
                })?;
                let projection =
                    planned(load, scope, WeightKind::PerLayerProjection, binding.projection)?;
                post_norm(
                    &mut terms,
                    projection,
                    planned(load, scope, WeightKind::PostNorm, norm)?,
                    binding.activation,
                )?;
            }
        }

        // The selection readout graph: final-norm features, the vocabulary
        // projection (its physical matrix, even when tied to the embedding),
        // and sampling of the one F32 logits row.
        let readout = target.readout();
        let norm = planned(
            load,
            WeightScope::Target,
            WeightKind::OutputNorm,
            readout.norm,
        )?;
        terms.add(
            MeasurementKey::readout_features(readout.norm, readout.activation),
            1,
            norm.resident_bytes,
        )?;
        let output = planned(
            load,
            WeightScope::Target,
            WeightKind::Output,
            readout.weight,
        )?;
        terms.project_whole(output, |element| {
            MeasurementKey::readout_head(readout.norm, element, readout.activation)
        })?;
        terms.add(
            MeasurementKey::sample_rows(),
            1,
            vocabulary
                .checked_mul(4)
                .ok_or_else(|| demand_error("logits bytes overflow"))?,
        )?;
        // Every entry call of the step depends on the one before it (the
        // step's graph runs are one dependency chain), and the step is
        // submitted once and waited on for its selection.
        let calls = terms
            .0
            .iter()
            .try_fold(0u64, |calls, term| calls.checked_add(term.launches))
            .ok_or_else(|| demand_error("entry call count overflows"))?;
        terms.add(MeasurementKey::launch_dependency(), calls, 0)?;
        terms.add(MeasurementKey::step_submission(), 1, 0)?;
        Ok(Self { terms: terms.0 })
    }

    /// The costs this demand's estimate needs that the basis does not hold:
    /// each term's cost key and weight format, unmeasured. A nonempty result
    /// makes the speed estimate unavailable.
    pub fn missing_costs(&self, basis: &MeasurementBasis) -> Vec<MeasurementKey> {
        let mut missing: Vec<MeasurementKey> = Vec::new();
        for term in &self.terms {
            let cost = term.key.cost();
            let format = term
                .weight()
                .map(|weight| MeasurementKey::weight_format(weight, cost.bindings[0]));
            for key in std::iter::once(cost).chain(format) {
                let measured = matches!(basis.get(&key), Some(ClassMeasurement::Measured { .. }));
                if !measured && !missing.contains(&key) {
                    missing.push(key);
                }
            }
        }
        missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::basis::OperationClass;
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

    fn term(demand: &DecodeDemand, class: OperationClass) -> &DemandTerm {
        let [term] = demand
            .terms
            .iter()
            .filter(|term| term.key.class == class)
            .collect::<Vec<_>>()[..]
        else {
            panic!("fixture has one {} term", class.name());
        };
        term
    }

    #[test]
    fn every_plain_decode_launch_is_a_term_with_planned_bytes() {
        let (definition, load) = fixture();
        let demand = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let classes = demand
            .terms
            .iter()
            .map(|term| term.key.class)
            .collect::<Vec<_>>();
        assert_eq!(
            classes,
            [
                OperationClass::EmbeddingRows,
                OperationClass::AttentionProject,
                OperationClass::AttentionDecode,
                OperationClass::AttentionOutput,
                OperationClass::DenseExpand,
                OperationClass::DenseOutput,
                OperationClass::ReadoutFeatures,
                OperationClass::ReadoutHead,
                OperationClass::SampleRows,
                OperationClass::LaunchDependency,
                OperationClass::StepSubmission,
            ]
        );
        // Nine entry calls, each depending on the previous; one step.
        assert_eq!(term(&demand, OperationClass::LaunchDependency).launches, 9);
        assert!(demand
            .terms
            .iter()
            .filter(|term| term.key.class != OperationClass::LaunchDependency)
            .all(|term| term.launches == 1));
        let bytes = |kinds: &[WeightKind]| {
            kinds
                .iter()
                .map(|kind| resident(&load, *kind).resident_bytes)
                .sum::<u64>()
        };
        // The fixture's weights share one representation, so each segmented
        // entry is one term.
        assert_eq!(
            term(&demand, OperationClass::AttentionProject).bytes,
            bytes(&[WeightKind::QueryGate, WeightKind::Key, WeightKind::Value])
        );
        assert_eq!(
            term(&demand, OperationClass::DenseExpand).bytes,
            bytes(&[WeightKind::DenseGate, WeightKind::DenseUp])
        );
        assert_eq!(
            term(&demand, OperationClass::DenseOutput).bytes,
            bytes(&[WeightKind::DenseDown])
        );
        assert_eq!(
            term(&demand, OperationClass::ReadoutHead).bytes,
            bytes(&[WeightKind::Output])
        );
        assert_eq!(
            term(&demand, OperationClass::SampleRows).bytes,
            definition.decoder.vocabulary * 4
        );
        let key = &term(&demand, OperationClass::DenseOutput).key;
        assert_eq!(
            *key,
            MeasurementKey::dense_output(
                resident(&load, WeightKind::DenseDown).resident,
                Element::bf16()
            )
        );
    }

    #[test]
    fn history_bytes_per_token_follow_the_state_layout_and_codec() {
        let (definition, load) = fixture();
        // One kv head of width 64: dense bf16 keys and values, 2 × 64 × 2.
        let dense = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let decode = term(&dense, OperationClass::AttentionDecode);
        assert_eq!((decode.bytes, decode.bytes_per_context_token), (0, 256));
        assert!(dense
            .terms
            .iter()
            .filter(|term| term.key.class != OperationClass::AttentionDecode)
            .all(|term| term.bytes_per_context_token == 0));
        // Affine: 64 key bytes, 32 value bytes and two (scale, zero) f16
        // pairs per 32-value group for each.
        let affine = DecodeDemand::from_model(&definition, &load, KvCodec::AffineK8V4).unwrap();
        let decode = term(&affine, OperationClass::AttentionDecodeK8V4);
        assert_eq!(decode.bytes_per_context_token, 64 + 32 + 2 * (2 * 2 * 2));
        assert!(matches!(
            DecodeDemand::from_model(&definition, &load, KvCodec::RotatedK4V4),
            Err(AssessmentError::Demand(_))
        ));
    }

    #[test]
    fn weight_terms_carry_their_launch_rows_and_representation() {
        let (definition, load) = fixture();
        let demand = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let rows = |kinds: &[WeightKind]| {
            kinds
                .iter()
                .map(|kind| resident(&load, *kind).shape[0])
                .sum::<u64>()
        };
        let projection = term(&demand, OperationClass::AttentionProject);
        assert_eq!(
            projection.shape,
            TermShape::Projection {
                weight: resident(&load, WeightKind::QueryGate).resident,
                launch_rows: rows(&[WeightKind::QueryGate, WeightKind::Key, WeightKind::Value]),
            }
        );
        let TermShape::Projection { launch_rows, .. } =
            term(&demand, OperationClass::ReadoutHead).shape
        else {
            panic!("the readout head streams a weight");
        };
        assert_eq!(launch_rows, definition.decoder.vocabulary);
        assert_eq!(
            term(&demand, OperationClass::AttentionDecode).shape,
            TermShape::Attention(HeadGeometry {
                kv_heads: 1,
                group: 2,
                width: 64,
            })
        );
        assert_eq!(term(&demand, OperationClass::SampleRows).shape, TermShape::Plain);
    }

    #[test]
    fn segments_cost_by_representation_at_their_launch_rows() {
        let mut terms = Terms::default();
        let (_, load) = fixture();
        let query = resident(&load, WeightKind::QueryGate);
        let mut value = resident(&load, WeightKind::Value).clone();
        value.resident = Element::stored("q6k", Layout::Rows16).unwrap();
        let key =
            |element| MeasurementKey::attention_project(Element::bf16(), element, Element::bf16());
        let segment = |weight, rows, bytes| Segment { weight, rows, bytes };
        let launch = [
            segment(query, 4, 10),
            segment(query, 8, 20),
            segment(&value, 4, 30),
        ];
        terms.project(&launch, key).unwrap();
        terms
            .project(&[segment(query, 1, 1), segment(&value, 1, 3)], key)
            .unwrap();
        // A second launch of the first shape accumulates with it.
        terms.project(&launch, key).unwrap();
        let projection = |weight: &WeightPlan, launch_rows| TermShape::Projection {
            weight: weight.resident,
            launch_rows,
        };
        let term = |key, launches, bytes, shape| DemandTerm {
            key,
            launches,
            bytes,
            bytes_per_context_token: 0,
            context_window: None,
            shape,
        };
        assert_eq!(
            terms.0,
            [
                term(key(query.resident), 2, 60, projection(query, 16)),
                term(key(value.resident), 0, 60, projection(&value, 16)),
                term(key(query.resident), 1, 1, projection(query, 2)),
                term(key(value.resident), 0, 3, projection(&value, 2)),
            ]
        );
    }

    #[test]
    fn a_selected_expert_segment_streams_its_share() {
        let (_, load) = fixture();
        let mut experts = resident(&load, WeightKind::DenseUp).clone();
        experts.shape = vec![8, 32, 64];
        experts.resident_bytes = 8 * 32 * 64 * 2;
        let selected = Segment::selected(&experts, 8, 2).unwrap();
        assert_eq!((selected.rows, selected.bytes), (64, 2 * 32 * 64 * 2));
        assert!(Segment::selected(&experts, 3, 1).is_err());
    }
}
