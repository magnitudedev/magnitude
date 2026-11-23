//! Tuning cases of the attention block: the normed segmented projection, the
//! fused attention entries of each history codec (`attention_decode` and
//! `attention_decode_k8v4` up to [`VERIFY_ROWS`] rows, `attention_prefill` and
//! `attention_prefill_k8v4` beyond) and the output projection, in the form
//! of the layers a case binds.
//!
//! Attention points cross the rows an entry serves with the served history
//! lengths. Each point is one request: its rows see `context` accepted
//! history rows and append their keys and values right after them, as a
//! decode step, a speculative verification or a prefill chunk does. The
//! points of one history length read one shared set of pseudo-random
//! history planes through views that end at their appended rows; those rows
//! are the case state restored before each validation run. No point reads
//! rows another point appends, so every configuration sees the same inputs
//! whatever the point order.

use super::cases::{packing_statics, projection_shape, scale_extent, scale_statics};
use super::{
    CaseState, EntryTuning, ModelInputs, PointShape, SlabView, TuningInputs, TuningLimits,
    across_sequences, history_placement, row_points, served_row_points, with_contexts,
};
use crate::operators;
use crate::operators::attention::graph::{
    query_projection_dimensions,
    rotary_amplitudes, rotary_components, rotary_frequencies, selected_segments,
};
use crate::programs::invocation_demand::Launch;
use magnitude_kernels::constants::VERIFY_ROWS;
use crate::operators::attention::{FreshRows, PickedSource, fresh_rows, picked_source, shared_rows};
use crate::{AttentionBinding, AttentionShape, LowRankQueryBinding, SelectionBinding};
use magnitude_family_contracts::{
    Attention, HeadHistory, HistoryDomain, Operator, WeightKind, WeightScope,
};
use magnitude_kernels::{
    attention_decode, attention_decode_k8v4, attention_output, attention_prefill,
    attention_prefill_k8v4, attention_project,
};
use seismic::{BuildError, Device, Element, Tensor};
use std::collections::BTreeMap;
use std::ops::Range;

/// `attention_project`: RMS prologue, one segmented query | gate | key |
/// value projection.
#[derive(Clone)]
pub(crate) struct AttentionProjectTuning {
    pub binding: AttentionBinding,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct AttentionProjectCase {
    hidden: Tensor,
    input_norm: Tensor,
    query: Tensor,
    gate: Tensor,
    key: Tensor,
    value: Tensor,
    indexer_query: Tensor,
    indexer_key: Tensor,
    epsilon: f32,
    /// The first four segments' scales: query, gate, key, value.
    scales: [Tensor; 4],
    /// The first four segments' packing bounds: query, gate, key, value.
    bounds: [Tensor; 4],
}

/// The unit scales of `attention_project`'s segments at `scope` (absent for
/// a segment it does not read, `None`), at their planned extents.
pub(super) fn segment_scales(
    inputs: &TuningInputs<'_, '_>,
    scope: WeightScope,
    kinds: [Option<WeightKind>; 4],
) -> Result<[Tensor; 4], BuildError> {
    let scale = |kind: Option<WeightKind>| match kind {
        Some(kind) => inputs.unit_scale(inputs.scale_extent(scope, kind)?),
        None => inputs.unit_scale(0),
    };
    Ok([
        scale(kinds[0])?,
        scale(kinds[1])?,
        scale(kinds[2])?,
        scale(kinds[3])?,
    ])
}

/// The packing bounds of `attention_project`'s segments at `scope` (absent
/// for a segment it does not read, `None`).
pub(super) fn segment_bounds(
    inputs: &mut TuningInputs<'_, '_>,
    scope: WeightScope,
    kinds: [Option<WeightKind>; 4],
) -> Result<[Tensor; 4], BuildError> {
    let mut bound = |kind: Option<WeightKind>| match kind {
        Some(kind) => inputs.bound(scope, kind),
        None => inputs.unit_scale(0),
    };
    Ok([
        bound(kinds[0])?,
        bound(kinds[1])?,
        bound(kinds[2])?,
        bound(kinds[3])?,
    ])
}

impl AttentionProjectTuning {
    /// The segments' weight kinds whose scales the projection binds, in
    /// `attention_project`'s order; none for an absent segment (and for the
    /// query, gate and indexer query of a key-value-only projection), as the
    /// block's graph binds them.
    fn operand_kinds(&self, query: WeightKind) -> [Option<WeightKind>; 4] {
        let shape = self.binding.shape;
        let projects = !self.binding.key_value_only;
        [
            projects.then_some(query),
            (projects && shape.gate_rows() > 0).then_some(WeightKind::AttentionGate),
            (shape.key_rows() > 0).then_some(WeightKind::Key),
            (shape.value_rows() > 0).then_some(WeightKind::Value),
        ]
    }

    fn elements(&self) -> attention_project::Elements {
        let b = self.binding;
        attention_project::Elements {
            NW: b.norm,
            QW: b.query_segment(),
            GW: b.gate_segment(),
            KW: b.key_segment(),
            VW: b.value_segment(),
            IQW: b.indexer_query_segment(),
            IKW: b.indexer_key_segment(),
            A: b.activation,
            Y: b.activation,
            IY: b.indexer_key_output(),
        }
    }
}

/// The attention operator of the layers a case binds.
fn operator<'i>(
    inputs: &'i ModelInputs<'_>,
    scopes: &[WeightScope],
) -> Result<&'i Attention, String> {
    match inputs.operator(scopes)? {
        Operator::Attention(attention) => Ok(attention),
        other => Err(format!("a {} layer has no attention", other.name())),
    }
}

impl EntryTuning for AttentionProjectTuning {
    type Entry = attention_project::Entry;
    type Case = AttentionProjectCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        let b = self.binding;
        format!(
            "NW={},QW={},GW={},KW={},VW={},IQW={},IKW={},A={},IY={}",
            b.norm.name(),
            b.query_segment().name(),
            b.gate_segment().name(),
            b.key_segment().name(),
            b.value_segment().name(),
            b.indexer_query_segment().name(),
            b.indexer_key_segment().name(),
            b.activation.name(),
            b.indexer_key_output().name(),
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        let query = operators::attention::query_segment_kind(operator(inputs, &self.scopes)?);
        let segment = self.binding.query_segment_rows();
        let [_, statics @ ..] = if self.binding.key_value_only {
            shape.key_value_dimensions(0)
        } else {
            shape.project_dimensions(0, segment)
        };
        if projection_shape(inputs, &self.scopes, query)? != (segment, shape.widths.hidden) {
            return Err("the query projection disagrees with the binding".into());
        }
        let mut statics = statics.to_vec();
        statics.extend([
            ("IQ", self.binding.indexer_query_rows()),
            ("IK", self.binding.indexer_key_rows()),
        ]);
        let [query, gate, key, value] = self.operand_kinds(query);
        statics.extend(scale_statics(
            inputs,
            &self.scopes,
            &[("QS", query), ("GS", gate), ("KS", key), ("VS", value)],
        )?);
        statics.extend(packing_statics(
            inputs,
            &self.scopes,
            &[("QB", query), ("GB", gate), ("KB", key), ("VB", value)],
        )?);
        let (parts, norm) = inputs.input_form(&self.scopes, shape.widths.hidden)?;
        statics.extend([("PT", parts), ("NM", norm)]);
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let shape = self.binding.shape;
        let query_kind = operators::attention::query_segment_kind(operator(inputs, &self.scopes)?);
        let (parts, norm) = inputs.input_form(&self.scopes, shape.widths.hidden)?;
        let iq_rows = self.binding.indexer_query_rows();
        let ik_rows = self.binding.indexer_key_rows();
        let kinds = self.operand_kinds(query_kind);
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let query = inputs.weight(scope, query_kind)?;
                // An absent segment reads zero rows of the query weight.
                let mut segment = |rows: u64, kind| {
                    if rows > 0 {
                        inputs.weight(scope, kind)
                    } else {
                        query.slice_leading(0, 0).map_err(BuildError::from)
                    }
                };
                Ok(AttentionProjectCase {
                    gate: {
                        let gate = segment(shape.gate_rows(), WeightKind::AttentionGate)?;
                        if self.binding.key_value_only {
                            gate.slice_leading(0, 0)
                                .map_err(|error| error.to_string())?
                        } else {
                            gate
                        }
                    },
                    key: segment(shape.key_rows(), WeightKind::Key)?,
                    value: segment(shape.value_rows(), WeightKind::Value)?,
                    indexer_query: segment(iq_rows, WeightKind::IndexerQuery)?,
                    indexer_key: segment(ik_rows, WeightKind::IndexerKey)?,
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, parts * shape.widths.hidden],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.input_norm(
                        scope,
                        WeightKind::InputNorm,
                        self.binding.norm,
                        norm,
                    )?,
                    query: if self.binding.key_value_only {
                        query
                            .slice_leading(0, 0)
                            .map_err(|error| error.to_string())?
                    } else {
                        query
                    },
                    epsilon: self.epsilon,
                    scales: segment_scales(inputs, scope, kinds)?,
                    bounds: segment_bounds(inputs, scope, kinds)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_project::Args<'a> {
        attention_project::Args {
            hidden: &case.hidden,
            input_norm: &case.input_norm,
            query_weight: &case.query,
            gate_weight: &case.gate,
            key_weight: &case.key,
            value_weight: &case.value,
            indexer_query_weight: &case.indexer_query,
            indexer_key_weight: &case.indexer_key,
            epsilon: case.epsilon,
            project_mode: 0,
            query_scale: &case.scales[0],
            gate_scale: &case.scales[1],
            key_scale: &case.scales[2],
            value_scale: &case.scales[3],
            query_bound: &case.bounds[0],
            gate_bound: &case.bounds[1],
            key_bound: &case.bounds[2],
            value_bound: &case.bounds[3],
        }
    }

    generated_entry!(attention_project, this => this.elements());
}

/// `attention_project` as a low-rank query's up projection: the F32 `qr`
/// rows (`rank` wide) normalized by their norm, then projected to the query
/// rows, with no gate, key or value rows.
#[derive(Clone)]
pub(crate) struct LowRankQueryProjectTuning {
    pub binding: AttentionBinding,
    pub low_rank: LowRankQueryBinding,
    pub scopes: Vec<WeightScope>,
}

impl LowRankQueryProjectTuning {
    /// The query weight's kind (`up`) as the only segment.
    fn operand_kinds(query: WeightKind) -> [Option<WeightKind>; 4] {
        [Some(query), None, None, None]
    }

    fn elements(&self) -> attention_project::Elements {
        let query = self.binding.query;
        attention_project::Elements {
            NW: self.low_rank.norm,
            QW: query,
            GW: query,
            KW: query,
            VW: query,
            IQW: query,
            IKW: query,
            A: self.binding.activation,
            Y: self.binding.activation,
            IY: self.binding.activation,
        }
    }
}

impl EntryTuning for LowRankQueryProjectTuning {
    type Entry = attention_project::Entry;
    type Case = AttentionProjectCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "NW={},QW={},A={};low-rank query",
            self.low_rank.norm.name(),
            self.binding.query.name(),
            self.binding.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let rank = self.low_rank.rank;
        let rows = self.binding.query_rows();
        let query = operators::attention::query_kind(operator(inputs, &self.scopes)?);
        if projection_shape(inputs, &self.scopes, query)? != (rows, rank) {
            return Err("the low-rank query's up projection disagrees with the binding".into());
        }
        let [_, statics @ ..] = query_projection_dimensions(0, rank, rows);
        let mut statics = [statics.as_slice(), &[("PT", 1), ("NM", rank)]].concat();
        let [query, gate, key, value] = Self::operand_kinds(query);
        statics.extend(scale_statics(
            inputs,
            &self.scopes,
            &[("QS", query), ("GS", gate), ("KS", key), ("VS", value)],
        )?);
        statics.extend(packing_statics(
            inputs,
            &self.scopes,
            &[("QB", query), ("GB", gate), ("KB", key), ("VB", value)],
        )?);
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let rank = self.low_rank.rank;
        let attention = operator(inputs, &self.scopes)?;
        let query_kind = operators::attention::query_kind(attention);
        let (_, epsilon) = operators::attention::low_rank_query(attention)
            .ok_or("the binding's low-rank query is a full query")?;
        let kinds = Self::operand_kinds(query_kind);
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let query = inputs.weight(scope, query_kind)?;
                let empty = query
                    .slice_leading(0, 0)
                    .map_err(|error| error.to_string())?;
                Ok(AttentionProjectCase {
                    gate: empty.clone(),
                    key: empty.clone(),
                    value: empty.clone(),
                    indexer_query: empty.clone(),
                    indexer_key: empty,
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, rank],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.weight(scope, WeightKind::QueryDownNorm)?,
                    query,
                    epsilon: epsilon as f32,
                    scales: segment_scales(inputs, scope, kinds)?,
                    bounds: segment_bounds(inputs, scope, kinds)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_project::Args<'a> {
        AttentionProjectTuning::args(case)
    }

    generated_entry!(attention_project, this => this.elements());
}

/// What one of an own index-list source's projections reads
/// (`operators::attention::pooled`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum PooledProjectInput {
    /// The layer's input `x`, in the sublayer's input form under its input
    /// norm (the `u | g` and head-weight projections).
    Hidden { hidden: u64 },
    /// The low-rank query's F32 `qr` rows under their norm (the indexer's
    /// query).
    QueryRank { rank: u64 },
    /// F32 rows of `width` under no norm (the indexer's keys, over the
    /// pooled latent).
    Plain { width: u64 },
    /// The tokens' F32 `u` rows of `width` under the pooled rows' norm (the
    /// indexer's keys at rate 1, whose latent is that normed row).
    Tokens { width: u64 },
}

/// `attention_project` as one of an indexing index-list source's
/// projections: `segments`, a query segment and (for `u | g`) a gate
/// segment, over `input`, at the elements
/// `operators::attention::pooled::PickedElements` assigns.
#[derive(Clone)]
pub(crate) struct PooledProjectTuning {
    pub elements: [(&'static str, Element); 10],
    pub segments: [Option<WeightKind>; 2],
    pub input: PooledProjectInput,
    pub scopes: Vec<WeightScope>,
    /// The input norm's epsilon, for an `x` input.
    pub epsilon: f32,
}

impl PooledProjectTuning {
    fn operand_kinds(&self) -> [Option<WeightKind>; 4] {
        [self.segments[0], self.segments[1], None, None]
    }

    fn element(&self, name: &str) -> Element {
        self.elements
            .iter()
            .find(|(element, _)| *element == name)
            .map(|(_, element)| *element)
            .expect("an attention_project element assignment names every element")
    }

    fn elements(&self) -> attention_project::Elements {
        attention_project::Elements {
            NW: self.element("NW"),
            QW: self.element("QW"),
            GW: self.element("GW"),
            KW: self.element("KW"),
            VW: self.element("VW"),
            IQW: self.element("IQW"),
            IKW: self.element("IKW"),
            A: self.element("A"),
            Y: self.element("Y"),
            IY: self.element("IY"),
        }
    }

    /// The input's width, parts and norm extent (`D`, `PT`, `NM`).
    fn form(&self, inputs: &ModelInputs<'_>) -> Result<(u64, u64, u64), String> {
        Ok(match self.input {
            PooledProjectInput::Hidden { hidden } => {
                let (parts, norm) = inputs.input_form(&self.scopes, hidden)?;
                (hidden, parts, norm)
            }
            PooledProjectInput::QueryRank { rank } => (rank, 1, rank),
            PooledProjectInput::Plain { width } => (width, 1, 0),
            PooledProjectInput::Tokens { width } => (width, 1, width),
        })
    }

    /// The input norm's weight and epsilon.
    fn norm(&self, inputs: &ModelInputs<'_>) -> Result<(WeightKind, f32), String> {
        Ok(match self.input {
            PooledProjectInput::Hidden { .. } => (WeightKind::InputNorm, self.epsilon),
            PooledProjectInput::QueryRank { .. } => {
                let (_, epsilon) = operators::attention::low_rank_query(operator(inputs, &self.scopes)?)
                    .ok_or("an indexer over the low-rank query of a full query")?;
                (WeightKind::QueryDownNorm, epsilon as f32)
            }
            // Zero rows: no kind is read.
            PooledProjectInput::Plain { .. } => (WeightKind::IndexerKeyNorm, 0.0),
            PooledProjectInput::Tokens { .. } => {
                let Ok(Some(operators::attention::PickedSource::Own { rows, .. })) =
                    operators::attention::picked_source(operator(inputs, &self.scopes)?)
                else {
                    return Err("a rate-1 key projection of a layer without its own pooled rows".into());
                };
                let norm = operators::attention::pooled_norm(rows).map_err(|error| error.to_string())?;
                (WeightKind::PooledNorm, norm.epsilon as f32)
            }
        })
    }

    /// The segments' rows (`Q`, `GR`), each checked against the input.
    fn rows(&self, inputs: &ModelInputs<'_>, width: u64) -> Result<[u64; 2], String> {
        let mut rows = [0; 2];
        for (rows, kind) in rows.iter_mut().zip(self.segments) {
            if let Some(kind) = kind {
                let (out, columns) = projection_shape(inputs, &self.scopes, kind)?;
                if columns != width {
                    return Err(format!("{kind:?} disagrees with its projection's input"));
                }
                *rows = out;
            }
        }
        Ok(rows)
    }
}

impl EntryTuning for PooledProjectTuning {
    type Entry = attention_project::Entry;
    type Case = AttentionProjectCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "NW={},QW={},GW={},A={};pooled {:?}",
            self.element("NW").name(),
            self.element("QW").name(),
            self.element("GW").name(),
            self.element("A").name(),
            self.segments,
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (width, parts, norm) = self.form(inputs)?;
        let [query_rows, gate_rows] = self.rows(inputs, width)?;
        let mut statics = vec![
            ("D", width),
            ("PT", parts),
            ("NM", norm),
            ("Q", query_rows),
            ("GR", gate_rows),
            ("K", 0),
            ("V", 0),
            ("IQ", 0),
            ("IK", 0),
        ];
        let [query, gate, key, value] = self.operand_kinds();
        statics.extend(scale_statics(
            inputs,
            &self.scopes,
            &[("QS", query), ("GS", gate), ("KS", key), ("VS", value)],
        )?);
        statics.extend(packing_statics(
            inputs,
            &self.scopes,
            &[("QB", query), ("GB", gate), ("KB", key), ("VB", value)],
        )?);
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let (width, parts, norm_extent) = self.form(inputs)?;
        let (norm, epsilon) = self.norm(inputs)?;
        let [query_kind, gate_kind] = self.segments;
        let query_kind = query_kind.ok_or("a pooled projection without its query segment")?;
        let norm_element = self.element("NW");
        let kinds = self.operand_kinds();
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let query = inputs.weight(scope, query_kind)?;
                let empty = query.slice_leading(0, 0).map_err(|error| error.to_string())?;
                Ok(AttentionProjectCase {
                    gate: match gate_kind {
                        Some(kind) => inputs.weight(scope, kind)?,
                        None => empty.clone(),
                    },
                    key: empty.clone(),
                    value: empty.clone(),
                    indexer_query: empty.clone(),
                    indexer_key: empty,
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, parts * width],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.input_norm(scope, norm, norm_element, norm_extent)?,
                    query,
                    epsilon,
                    scales: segment_scales(inputs, scope, kinds)?,
                    bounds: segment_bounds(inputs, scope, kinds)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_project::Args<'a> {
        AttentionProjectTuning::args(case)
    }

    generated_entry!(attention_project, this => this.elements());
}

/// `attention_output`: output projection plus residual.
#[derive(Clone)]
pub(crate) struct AttentionOutputTuning {
    pub output: Element,
    pub activation: Element,
    pub shape: AttentionShape,
    /// The projection's input rows, `(Q, W)` (`AttentionBinding::output_input`).
    pub input: (u64, u64),
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct AttentionOutputCase {
    hidden: Tensor,
    inject: Tensor,
    gated: Tensor,
    output: Tensor,
    scale: Tensor,
    bound: Tensor,
}

impl AttentionOutputTuning {
    fn elements(&self) -> attention_output::Elements {
        attention_output::Elements {
            OW: self.output,
            A: self.activation,
        }
    }
}

impl EntryTuning for AttentionOutputTuning {
    type Entry = attention_output::Entry;
    type Case = AttentionOutputCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!("OW={},A={}", self.output.name(), self.activation.name())
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let (q, w) = self.input;
        if projection_shape(inputs, &self.scopes, WeightKind::AttentionOutput)?
            != (shape.widths.hidden, q * w)
        {
            return Err("the attention output projection disagrees with the binding".into());
        }
        let mut statics = vec![
            ("D", shape.widths.hidden),
            ("Q", q),
            ("W", w),
            (
                "OS",
                scale_extent(inputs, &self.scopes, WeightKind::AttentionOutput)?,
            ),
        ];
        statics.extend(packing_statics(
            inputs,
            &self.scopes,
            &[("OB", Some(WeightKind::AttentionOutput))],
        )?);
        let (streams, inject) = inputs.join(&self.scopes)?;
        statics.extend([("SN", streams), ("IN", inject)]);
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let shape = self.shape;
        let (streams, inject) = inputs.join(&self.scopes)?;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 2 * index as u64;
                Ok(AttentionOutputCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, streams * shape.widths.hidden],
                        seed + 1,
                    )?,
                    inject: inputs.activation(Element::f32(), &[point.rows, inject], seed + 3)?,
                    gated: inputs.activation(
                        self.activation,
                        &[point.rows, self.input.0, self.input.1],
                        seed + 2,
                    )?,
                    output: inputs.weight(scope, WeightKind::AttentionOutput)?,
                    scale: inputs
                        .unit_scale(inputs.scale_extent(scope, WeightKind::AttentionOutput)?)?,
                    bound: inputs.bound(scope, WeightKind::AttentionOutput)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_output::Args<'a> {
        attention_output::Args {
            hidden: &case.hidden,
            inject: &case.inject,
            gated: &case.gated,
            output_weight: &case.output,
            output_scale: &case.scale,
            output_bound: &case.bound,
        }
    }

    generated_entry!(attention_output, this => this.elements());
}

/// What every fused attention entry tunes over.
#[derive(Clone)]
pub(crate) struct AttentionMix {
    pub codec: magnitude_state::CodecSpec,
    pub activation: Element,
    pub shape: AttentionShape,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
    /// For the affine prefill: which of the layers' prefill launches the
    /// kernel serves, and whether it lists the history row tiles their rows
    /// see (the entry's `L`).
    pub listing: PrefillListing,
    /// The most history rows the layers the case binds keep
    /// ([`history_window`]): its points see no longer history.
    pub window: Option<u64>,
    /// A selected read's indexer: the prefill then reads through its
    /// WorkList bitmap (`B` words, static), tuned at the indexer's `top`
    /// block density plus each row's tail.
    pub selection: Option<SelectionBinding>,
    /// The index-list source the layers' picks read ([`picked_rows`]):
    /// present exactly when the shape picks rows (`KS > 0`).
    pub picked: Option<PickedRows>,
}

impl AttentionMix {
    /// The prefill's WorkList bitmap words `B`: 0 for a read of every
    /// visible row.
    fn words(&self) -> u64 {
        self.selection.map_or(0, |selection| selection.shape.words)
    }

    /// The keys `BR` of a WorkList bitmap's block: 1 for a read of every
    /// visible row.
    fn block_keys(&self) -> u64 {
        self.selection.map_or(1, |selection| selection.shape.block)
    }

    /// The prefill's WorkList selection bitmap (`[rows, B]`), matching
    /// `unit_reads`: a row reads every unit when its complete units are at
    /// most `top`; otherwise `top` complete units spaced by `complete / top`
    /// (so consecutive rows in a query tile share their selected blocks) plus
    /// every unit from its tail (`complete`) on.
    fn selection_bitmap(&self, row_keys: &[u64]) -> Vec<u32> {
        let Some(selection) = self.selection else {
            return Vec::new();
        };
        let words = selection.shape.words as usize;
        if words == 0 {
            return Vec::new();
        }
        let block = selection.shape.block;
        let top = selection.indexer.top;
        let total_units = selection.shape.units();
        let mut bitmap = vec![0u32; row_keys.len() * words];
        for (&keys, row_words) in row_keys.iter().zip(bitmap.chunks_exact_mut(words)) {
            let complete = (keys / block).min(total_units);
            if complete <= top {
                row_words.fill(u32::MAX);
                continue;
            }
            if top > 0 {
                let step = complete / top;
                for pick in 0..top {
                    let unit = (pick * step) as usize;
                    row_words[unit / 32] |= 1u32 << (unit % 32);
                }
            }
            if complete < total_units {
                let first_word = (complete / 32) as usize;
                let first_bit = (complete % 32) as u32;
                row_words[first_word] |= u32::MAX << first_bit;
                row_words[first_word + 1..].fill(u32::MAX);
            }
        }
        bitmap
    }
}

/// Which prefill launches of its layers an affine prefill kernel serves.
/// Where the layers' graphs list the history row tiles a launch's rows see
/// (`StateResourcePlan::lists_history_tiles`), a launch whose rows see at
/// most one request's history takes the listed kernel and any other the
/// unlisted one (`listed_tiles` in the target's step): the two are
/// alternatives for the same layers, each with its own admissible forms
/// (`L` is static where the forms differ by it), and a launch is demand of
/// the one that serves it alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefillListing {
    /// Every launch, listing none: the layers' graphs list no tiles.
    Every,
    /// The launches whose rows see at most `capacity` history rows, one
    /// request's, listing the tiles they see.
    Listed { capacity: u64 },
    /// The launches whose rows see more than `capacity` history rows,
    /// listing none.
    Unlisted { capacity: u64 },
}

impl PrefillListing {
    /// Whether the kernel lists its rows' history row tiles.
    pub fn lists(self) -> bool {
        matches!(self, Self::Listed { .. })
    }

    /// Whether the kernel serves `launch`: by the history rows its rows
    /// see, each sequence's history and the rows it writes.
    pub fn serves(self, launch: &Launch) -> bool {
        let seen = launch.sequences * launch.history + launch.rows;
        match self {
            Self::Every => true,
            Self::Listed { capacity } => seen <= capacity,
            Self::Unlisted { capacity } => seen > capacity,
        }
    }
}

/// The most history rows the layers `scopes` name keep: the largest window
/// among them, none when any keeps the whole context. A Shared layer reads
/// its source's history and keeps what the source does; a head reading the
/// target's history names a target layer as its source.
pub(crate) fn history_window(
    inputs: &ModelInputs<'_>,
    scopes: &[WeightScope],
) -> Result<Option<u64>, String> {
    let mut windows = Vec::with_capacity(scopes.len());
    for scope in scopes {
        let domain = match fresh_rows(operator(inputs, &[*scope])?) {
            FreshRows::Own(rows) => HistoryDomain::from(rows.domain),
            FreshRows::Shared(source) => {
                let target_read = matches!(scope, WeightScope::HeadSublayer(_))
                    && inputs
                        .definition
                        .head
                        .as_ref()
                        .is_some_and(|head| head.history == HeadHistory::Target);
                if target_read {
                    // Through the target's own sharing to the owning layer.
                    inputs
                        .definition
                        .decoder
                        .history_domain(source)
                        .ok_or_else(|| format!("{scope:?} reads no target history"))?
                } else {
                    let owner = match scope {
                        WeightScope::TargetSublayer(_) => WeightScope::TargetSublayer(source),
                        WeightScope::HeadSublayer(_) => WeightScope::HeadSublayer(source),
                        WeightScope::DraftSublayer(_) => WeightScope::DraftSublayer(source),
                        other => return Err(format!("{other:?} names no sublayer")),
                    };
                    match fresh_rows(operator(inputs, &[owner])?) {
                        FreshRows::Own(rows) => HistoryDomain::from(rows.domain),
                        FreshRows::Shared(_) => {
                            return Err(format!("{owner:?} owns no history for {scope:?} to read"));
                        }
                    }
                }
            }
        };
        match domain {
            HistoryDomain::Window { tokens } => windows.push(tokens),
            HistoryDomain::Token | HistoryDomain::Block { .. } => return Ok(None),
        }
    }
    windows
        .into_iter()
        .max()
        .map(Some)
        .ok_or_else(|| "a tuning case needs at least one layer".to_owned())
}

/// The pooled rows an index-list source holds, as a unit's picks read them:
/// one row per `rate` tokens, up to the `context` limit, whatever window
/// the layers' first source keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PickedRows {
    pub rate: u64,
    pub context: u64,
}

impl PickedRows {
    /// The position the first row of a point with `context` history rows
    /// of layers keeping at most `window` stands for: `context`, but the
    /// longest history of a window shorter than the context limit stands
    /// for every longer prompt ([`with_contexts`]), whose pooled rows grow
    /// to the limit.
    fn reach(&self, window: Option<u64>, context: u64) -> u64 {
        match window {
            Some(window) if window < self.context && context >= window => self.context,
            _ => context,
        }
    }

    /// The pooled rows a row at `position` lists (within the context
    /// limit): the groups complete through it, as the store's Block domain
    /// holds them ([`magnitude_state::HistoryDomainKind::checkpoint_rows`],
    /// `(t + 1) / rate`).
    fn held(&self, position: u64) -> u64 {
        let through = (position + 1).min(self.context) as usize;
        magnitude_state::HistoryDomainKind::Block { rate: self.rate as usize }
            .checkpoint_rows(through)
            .expect("a Block domain references its complete groups") as u64
    }

    /// Pick `pick` of the `picks` of a row of sequence `sequence` of
    /// `sequences` whose source holds `held` rows: each held row while they
    /// are no more than the picks, then none (-1, as an indexer fills a
    /// short list), or else the picks spread evenly over them, the
    /// sequences' interleaved so that, where the rows allow, each reads
    /// rows of its own.
    fn pick(pick: u64, picks: u64, sequence: u64, sequences: u64, held: u64) -> i32 {
        let row = match held <= picks {
            true if pick < held => pick,
            true => return -1,
            false => (pick * sequences + sequence) * held / (picks * sequences),
        };
        i32::try_from(row).expect("a tuning plane's rows fit i32")
    }
}

/// The index-list source the layers at `scopes` read, up to the load's
/// context: the pooled rows of the fewest tokens per row among theirs,
/// which a step holds the most of (as [`history_window`] keeps the longest
/// window); none for layers reading one source.
pub(crate) fn picked_rows(
    inputs: &ModelInputs<'_>,
    scopes: &[WeightScope],
) -> Result<Option<PickedRows>, String> {
    let decoder = &inputs.definition.decoder;
    let mut rates = Vec::with_capacity(scopes.len());
    for scope in scopes {
        let rows = match picked_source(operator(inputs, &[*scope])?).map_err(|e| e.to_string())? {
            None => continue,
            Some(PickedSource::Own { rows, .. }) => rows,
            Some(
                PickedSource::Reindex { source, part, .. }
                | PickedSource::Reused { source, part, .. },
            ) => shared_rows(decoder, source, part).map_err(|e| e.to_string())?,
        };
        rates.push(rows.rate);
    }
    Ok(rates.into_iter().min().map(|rate| PickedRows {
        rate,
        context: inputs.limits.context_tokens,
    }))
}

/// `attention_decode`, for row classes up to [`VERIFY_ROWS`].
#[derive(Clone)]
pub(crate) struct AttentionDecodeTuning(pub AttentionMix);

/// `attention_prefill`, for the row classes `.1` names.
#[derive(Clone)]
pub(crate) struct AttentionPrefillTuning(pub AttentionMix, pub PrefillRows);

/// The row classes a dense prefill unit serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefillRows {
    /// The classes past the decode rows.
    Prefill,
    /// Every decode row class, where the decode entry holds no form of the
    /// shape (`graph::dense_decodes`).
    Decode,
}

/// `attention_decode_k8v4`, for row classes up to [`VERIFY_ROWS`].
#[derive(Clone)]
pub(crate) struct AttentionDecodeK8V4Tuning(pub AttentionMix);

/// `attention_prefill_k8v4`, for row classes beyond [`VERIFY_ROWS`].
#[derive(Clone)]
pub(crate) struct AttentionPrefillK8V4Tuning(pub AttentionMix);

/// One argument set of a fused entry: the inputs every codec shares, and the
/// history planes of the entry's codec.
pub(crate) struct AttentionMixCase<H> {
    query: Tensor,
    gate: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    value_norm: Tensor,
    /// The layer's sink logits `[SK, heads]` (the dense entries').
    sinks: Tensor,
    rotary_components: Tensor,
    rotary_frequencies: Tensor,
    rotary_amplitudes: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
    /// The history row tiles the case's rows see (the affine prefill's
    /// `history_tiles`).
    history_tiles: Tensor,
    /// The prefill's WorkList bitmap (`[rows, B]`, `top` complete blocks plus
    /// each row's tail; none in the Visible form) and its blocks' keys
    /// (`[0, BR]`, BR 1 in the Visible form).
    selection: Tensor,
    block_slots: Tensor,
    /// The dense entries' index-list source: each row's picks (`[rows,
    /// KS]`) of the rows its source holds, in a plane of the most any row
    /// holds (`[TS, KV, W]`, a slab port's), and source 0's key rounding.
    selected_rows: Tensor,
    selected: SlabView,
    key_rounding: crate::RoundingArgs,
    /// The rotary layout, output rotary, stored-key values and, for the
    /// dense entries, key rounding (`[0, RA, RT, RO, VK(, KR)]`, no values).
    form: Tensor,
    history: H,
    slab_rows: u32,
    epsilon: f32,
    scale: f32,
    gate_function: i32,
}

/// The history planes of one codec, as case state: views of the planes
/// shared by every point with the same history rows (one history length, in
/// any unit of the tuning), ending at a point's appended rows.
pub(crate) trait MixHistory: Sized {
    /// The codec of the store whose histories these planes are.
    const CODEC: magnitude_state::KvCodec;
    /// Planes of `rows` history rows, viewed up to `view` rows with
    /// `appended` written.
    fn build(
        inputs: &mut TuningInputs<'_, '_>,
        mix: &AttentionMix,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, BuildError>;
    fn share(&self) -> Self;
    /// The planes, by the entry parameter each binds.
    fn states(&self) -> Vec<(&'static str, &CaseState)>;
}

/// Dense key and value planes `[T, KV, W]` in the activation dtype; a
/// stored-key history (`VK`) has the key plane alone, and its entries bind
/// zero rows of a value plane.
pub(crate) struct DenseMixHistory {
    key: CaseState,
    value: DenseMixValues,
}

enum DenseMixValues {
    Own(CaseState),
    /// `[0, KV, W]`, never read or written.
    Key(Tensor),
}

impl DenseMixValues {
    /// The tensor the entries' `history_value` binds.
    fn tensor_mut(&mut self) -> &mut Tensor {
        match self {
            Self::Own(state) => state.tensor_mut(),
            Self::Key(absent) => absent,
        }
    }
}

impl MixHistory for DenseMixHistory {
    const CODEC: magnitude_state::KvCodec = magnitude_state::KvCodec::Dense;

    fn build(
        inputs: &mut TuningInputs<'_, '_>,
        mix: &AttentionMix,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, BuildError> {
        let shape = mix.shape;
        let mut plane = |name: &str, seed: u64| {
            let extents = [rows, shape.widths.kv_heads, shape.widths.width];
            let plane = inputs.shared(
                format!("{name}-{}-{extents:?}", mix.activation.name()),
                |inputs| inputs.activation(mix.activation, &extents, seed),
            )?;
            inputs.slab_state(plane, view, appended.clone())
        };
        let key = plane("history_key", 0x6b)?;
        let value = if shape.stored_key() == 1 {
            DenseMixValues::Key(inputs.activation(
                mix.activation,
                &[0, shape.widths.kv_heads, shape.widths.width],
                0x76,
            )?)
        } else {
            DenseMixValues::Own(plane("history_value", 0x76)?)
        };
        Ok(Self { key, value })
    }

    fn share(&self) -> Self {
        Self {
            key: self.key.share(),
            value: match &self.value {
                DenseMixValues::Own(state) => DenseMixValues::Own(state.share()),
                DenseMixValues::Key(absent) => DenseMixValues::Key(absent.clone()),
            },
        }
    }

    fn states(&self) -> Vec<(&'static str, &CaseState)> {
        let mut states = vec![("history_key", &self.key)];
        if let DenseMixValues::Own(value) = &self.value {
            states.push(("history_value", value));
        }
        states
    }
}

/// Affine K8/V4 planes: code rows `[T, KV, W * B / 32]` u32 and group
/// (scale, zero) pairs `[T, KV, 2 * W / group]` f16 per vector kind. Codes are pseudo-random and
/// every pair decodes its codes into [-1, 1], as the dense planes hold.
pub(crate) struct AffineMixHistory {
    key_codes: CaseState,
    key_coefficients: CaseState,
    value_codes: CaseState,
    value_coefficients: CaseState,
}

impl AffineMixHistory {
    fn codes(
        inputs: &TuningInputs<'_, '_>,
        extents: &[u64],
        seed: u64,
    ) -> Result<Tensor, BuildError> {
        inputs.generated(Element::u32(), extents, |count| {
            let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            Ok((0..count)
                .flat_map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 32) as u32).to_le_bytes()
                })
                .collect())
        })
    }

    /// Vary each affine group so an incorrect coefficient stride cannot hide.
    fn coefficients(
        inputs: &TuningInputs<'_, '_>,
        extents: &[u64],
        levels: u32,
    ) -> Result<Tensor, BuildError> {
        inputs.generated(Element::f16(), extents, |count| {
            Ok((0..count / 2)
                .flat_map(|index| {
                    let magnitude = 0.25 + (index.wrapping_mul(17) % 127) as f32 / 64.0;
                    let offset = -0.75 + (index.wrapping_mul(43) % 97) as f32 / 96.0;
                    [
                        seismic::f16_bits(magnitude / levels as f32).to_le_bytes(),
                        seismic::f16_bits(offset).to_le_bytes(),
                    ]
                })
                .flatten()
                .collect())
        })
    }
}

impl MixHistory for AffineMixHistory {
    const CODEC: magnitude_state::KvCodec = magnitude_state::KvCodec::AffineK8V4;

    fn build(
        inputs: &mut TuningInputs<'_, '_>,
        mix: &AttentionMix,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, BuildError> {
        let shape = mix.shape;
        let mut plane =
            |name: &str, build: &dyn Fn(&TuningInputs<'_, '_>) -> Result<Tensor, BuildError>| {
                let plane = inputs.shared(
                    format!(
                        "{name}-affine-{rows}x{}x{}-{:?}",
                        shape.widths.kv_heads, shape.widths.width, mix.codec
                    ),
                    |inputs| build(inputs),
                )?;
                inputs.slab_state(plane, view, appended.clone())
            };
        let geometry = mix.codec.affine_geometry().map_err(|e| e.to_string())?;
        let pairs = [rows, shape.widths.kv_heads, geometry.key_coefficients as u64];
        let value_pairs = [rows, shape.widths.kv_heads, geometry.value_coefficients as u64];
        Ok(Self {
            key_codes: plane("history_key_codes", &|inputs| {
                Self::codes(
                    inputs,
                    &[rows, shape.widths.kv_heads, geometry.key_codes as u64],
                    0x6b,
                )
            })?,
            key_coefficients: plane("history_key_coefficients", &|inputs| {
                Self::coefficients(inputs, &pairs, (1 << geometry.key_bits) - 1)
            })?,
            value_codes: plane("history_value_codes", &|inputs| {
                Self::codes(
                    inputs,
                    &[rows, shape.widths.kv_heads, geometry.value_codes as u64],
                    0x76,
                )
            })?,
            value_coefficients: plane("history_value_coefficients", &|inputs| {
                Self::coefficients(inputs, &value_pairs, (1 << geometry.value_bits) - 1)
            })?,
        })
    }

    fn share(&self) -> Self {
        Self {
            key_codes: self.key_codes.share(),
            key_coefficients: self.key_coefficients.share(),
            value_codes: self.value_codes.share(),
            value_coefficients: self.value_coefficients.share(),
        }
    }

    fn states(&self) -> Vec<(&'static str, &CaseState)> {
        vec![
            ("history_key_codes", &self.key_codes),
            ("history_key_coefficients", &self.key_coefficients),
            ("history_value_codes", &self.value_codes),
            ("history_value_coefficients", &self.value_coefficients),
        ]
    }
}

impl AttentionMix {
    /// The index-list source the layers' picks read; none without picks.
    fn index_list(&self) -> Option<PickedRows> {
        match (self.shape.widths.picks, self.picked) {
            (0, None) => None,
            (picks, Some(picked)) if picks > 0 => Some(picked),
            _ => unreachable!("a unit's layers pick rows exactly when they read an index-list source"),
        }
    }

    /// The rows `TS` of the index-list source's plane a case of `point`
    /// binds: what the source holds at the furthest row any of its
    /// sequences reaches; none without picks.
    fn picked_plane_rows(&self, point: &PointShape) -> u64 {
        self.index_list().map_or(0, |picked| {
            let reach = picked.reach(self.window, point.context.unwrap_or(0));
            picked.held(reach + point.rows - 1)
        })
    }

    /// The picks (`[rows, KS]`) of the rows of `point` at `coordinates`, of
    /// the sequences `row_slots`: each row's from the pooled rows its
    /// sequence holds at the position it stands for.
    fn picks(&self, point: &PointShape, coordinates: &[[i32; 4]], row_slots: &[i32]) -> Vec<i32> {
        let Some(picked) = self.index_list() else {
            return Vec::new();
        };
        let context = point.context.unwrap_or(0);
        let reach = picked.reach(self.window, context);
        let picks = self.shape.widths.picks;
        coordinates
            .iter()
            .zip(row_slots)
            .flat_map(|(coordinates, &sequence)| {
                let held = picked.held(reach + coordinates[0] as u64 - context);
                (0..picks).map(move |pick| {
                    PickedRows::pick(pick, picks, sequence as u64, point.sequences, held)
                })
            })
            .collect()
    }

    /// Whether the mix is a dense history's, whose entries also take source
    /// 0's key rounding and an index-list source.
    fn dense(&self) -> bool {
        self.codec.encoding() == magnitude_state::HistoryEncoding::Dense
    }

    /// The dense entries' `KR` and `KS` statics; none for the affine ones.
    fn dense_statics(&self) -> Vec<(&'static str, u64)> {
        match self.dense() {
            true => vec![("KR", self.shape.rounds_keys()), ("KS", self.shape.widths.picks)],
            false => Vec::new(),
        }
    }

    /// The dimensions of the unit's calls at `shapes` over every history
    /// segment class a store seals graphs for: a tuning point's rows see one
    /// span, a served launch's up to the class of
    /// [`magnitude_state::MAX_HISTORY_SPANS`], and the attend launches'
    /// threadgroup memory grows with the spans they schedule.
    fn served_dimensions(&self, shapes: &[PointShape]) -> Vec<BTreeMap<String, u64>> {
        // A selected read's decode rows read their units' segments of
        // each class's spans.
        let spans = std::iter::successors(Some(1u64), |segments| segments.checked_mul(2))
            .take_while(|segments| {
                *segments <= magnitude_state::MAX_HISTORY_SPANS.next_power_of_two() as u64
            })
            .collect::<Vec<_>>();
        let selected = self.selection.into_iter().flat_map(|selection| {
            spans
                .iter()
                .map(move |&segments| (selected_segments(selection.indexer.top, segments), true))
        });
        let classes = spans
            .iter()
            .map(|&segments| (segments, false))
            .chain(selected)
            .collect::<Vec<_>>();
        classes
            .into_iter()
            .flat_map(|(segments, selected)| {
                shapes.iter().filter_map(move |point| {
                    if selected && point.rows > VERIFY_ROWS {
                        return None;
                    }
                    let context = point.context?;
                    // A listed launch lists the tiles its rows see, one at
                    // least; an unlisted one passes an empty list of one
                    // column.
                    let tiles = if self.listing.lists() {
                        context.div_ceil(256).max(1)
                    } else {
                        1
                    };
                    Some(
                        self.shape
                            .mix_dimensions(point.rows, context + point.rows, segments)
                            .into_iter()
                            .chain([("L", u64::from(self.listing.lists())), ("HT", tiles)])
                            .chain(
                                if self.codec.encoding() == magnitude_state::HistoryEncoding::Affine
                                {
                                    crate::operators::attention::graph::affine_dimensions(
                                        self.codec,
                                    )
                                    .expect("validated affine descriptor")
                                } else {
                                    Vec::new()
                                },
                            )
                            // A case's picks read the rows its source
                            // holds.
                            .chain(self.dense_statics())
                            .chain(self.dense().then(|| ("TS", self.picked_plane_rows(point))))
                            .map(|(name, value)| (name.to_owned(), value))
                            .collect(),
                    )
                })
            })
            .collect()
    }

    fn bindings(&self) -> String {
        match () {
            _ if self.listing.lists() => format!("A={},L=1", self.activation.name()),
            _ if self.selection.is_some() => {
                format!(
                    "A={},B={},BR={}",
                    self.activation.name(),
                    self.words(),
                    self.block_keys()
                )
            }
            _ => format!("A={}", self.activation.name()),
        }
    }

    fn statics(&self) -> Vec<(&'static str, u64)> {
        crate::operators::attention::graph::form_statics(&self.shape, self.codec)
            .expect("a mix's codec is validated")
            .into_iter()
            .chain([("L", u64::from(self.listing.lists()))])
            .collect()
    }

    /// The history rows the planes of points with `context` history rows in
    /// each of `sequences` sequences hold: the histories, `histories` rows
    /// as they are laid out, plus the most rows any such point appends.
    fn history_rows(points: &[PointShape], context: u64, sequences: u64, histories: u64) -> u64 {
        histories
            + points
                .iter()
                .filter(|point| point.context == Some(context) && point.sequences == sequences)
                .map(|point| point.rows)
                .max()
                .unwrap_or(0)
    }

    fn rotation<H: MixHistory>(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
        points: Vec<PointShape>,
    ) -> Result<Vec<AttentionMixCase<H>>, BuildError> {
        let shape = self.shape;
        let rows = point.rows;
        let context = point
            .context
            .ok_or("attention tuning points carry a history length")?;
        // Points of one history length and sequence count share planes: they
        // all read their sequences' first `context` rows, which nothing
        // writes, and append after them. Another history length or sequence
        // count has its own planes, so no point ever reads rows another
        // point appended.
        let sequences = point.sequences;
        let placement = history_placement(inputs, H::CODEC, self.window, point)?;
        let histories = placement.extent;
        let view = histories + rows;
        let history = H::build(
            inputs,
            self,
            Self::history_rows(&points, context, sequences, histories),
            view,
            histories..view,
        )?;
        let tables = inputs.batch(rows, context, &placement.sequences, histories)?;
        let segments = tables.class.segments() as u64;
        let attention = operator(inputs, &self.scopes)?;
        let components = rotary_components(&attention.rotary)?;
        let frequencies = rotary_frequencies(&attention.rotary);
        let amplitudes = rotary_amplitudes(&attention.rotary);
        let scale = attention.scale as f32;
        let gate_function = operators::attention::gate_function(attention);
        let pairs = components.len() as u64;
        let coordinates = tables
            .coordinates
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        // The tuning batch has one Token history domain.
        let history_tables = &tables.histories[0];
        let visible = history_tables
            .visible
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let fresh = history_tables
            .fresh
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let (heads, width) = (shape.heads(), shape.widths.width);
        let fresh_rows = [u64::from(shape.form.is_fresh()), rows, shape.widths.kv_heads * width];
        let selected_rows = self.picks(
            point,
            &tables.coordinates[..rows as usize],
            &tables.row_slots[..rows as usize],
        );
        let plane_rows = self.picked_plane_rows(point);
        let selection_bits = if self.selection.is_some() {
            let row_keys = history_tables
                .visible
                .iter()
                .zip(&history_tables.fresh)
                .map(|(spans, &[fresh_lo, fresh_hi])| {
                    spans
                        .iter()
                        .map(|&[lo, hi]| (hi - lo).max(0) as u64)
                        .sum::<u64>()
                        + (fresh_hi - fresh_lo).max(0) as u64
                })
                .collect::<Vec<_>>();
            self.selection_bitmap(&row_keys)
        } else {
            Vec::new()
        };
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 4 * index as u64;
                // Head norms are the layers' own weights, or none.
                let mut head_norm = |present: bool, kind| {
                    if present {
                        inputs
                            .weight(scope, kind)?
                            .reshape(&[1, width])
                            .map_err(BuildError::from)
                    } else {
                        inputs.f32s(&[0, width], &[])
                    }
                };
                let query_norm = head_norm(shape.form.query_norm, WeightKind::QueryNorm)?;
                // A Shared layer's unread key norm port takes its query norm
                // (`graph::attention`).
                let key_norm = if !shape.form.is_fresh() {
                    query_norm.clone()
                } else {
                    head_norm(shape.form.key_norm, WeightKind::KeyNorm)?
                };
                // Sinks are the layers' own logits, or none.
                let sinks = if shape.form.sinks {
                    inputs
                        .weight(scope, WeightKind::AttentionSinks)?
                        .reshape(&[1, heads])
                        .map_err(|error| error.to_string())?
                } else {
                    inputs.f32s(&[0, heads], &[])?
                };
                Ok(AttentionMixCase {
                    query_norm,
                    key_norm,
                    value_norm: inputs.f32s(
                        &[u64::from(shape.form.value_norm), width],
                        &vec![1.0; (u64::from(shape.form.value_norm) * width) as usize],
                    )?,
                    sinks,
                    query: inputs.activation(
                        self.activation,
                        &[rows, heads, width + shape.interleaved_gate()],
                        seed + 1,
                    )?,
                    gate: inputs.activation(
                        self.activation,
                        &[rows, heads, shape.separate_gate()],
                        seed + 2,
                    )?,
                    key: inputs.activation(self.activation, &fresh_rows, seed + 3)?,
                    // Stored-key values take no value of their own (`VK`).
                    value: inputs.activation(
                        self.activation,
                        &[
                            u64::from(shape.form.is_fresh() && !shape.form.stored_key()),
                            rows,
                            shape.widths.kv_heads * width,
                        ],
                        seed + 4,
                    )?,
                    rotary_components: inputs.i32s(&[pairs], &components)?,
                    rotary_frequencies: inputs.f32s(&[pairs], &frequencies)?,
                    rotary_amplitudes: inputs.f32s(&[pairs], &amplitudes)?,
                    coordinates: inputs.i32s(&[rows, 4], &coordinates)?,
                    visible: inputs.i32s(&[rows, segments, 2], &visible)?,
                    fresh: inputs.i32s(&[rows, 2], &fresh)?,
                    destinations: inputs.i32s(&[rows], &history_tables.destinations)?,
                    history_tiles: {
                        // Exactly the tiles the case's rows see (one entry
                        // when they see none).
                        let spans = || visible.chunks_exact(2).map(|span| [span[0], span[1]]);
                        let bound = spans()
                            .map(|[start, end]| (end - start).max(0) as usize / 256 + 2)
                            .sum::<usize>();
                        let mut tiles = magnitude_batching::history_tiles(spans(), bound)
                            .ok_or("tuning history tiles exceed their bound")?;
                        tiles.truncate(tiles.iter().filter(|tile| **tile >= 0).count().max(1));
                        // An unlisted launch passes no list.
                        if !self.listing.lists() {
                            inputs.i32s(&[0, 1], &[])?
                        } else {
                            inputs.i32s(&[1, tiles.len() as u64], &tiles)?
                        }
                    },
                    selection: inputs.u32s(&[rows, self.words()], &selection_bits)?,
                    block_slots: inputs.i32s(&[0, self.block_keys()], &[])?,
                    selected_rows: inputs.i32s(&[rows, shape.widths.picks], &selected_rows)?,
                    selected: inputs.slab_view(inputs.activation(
                        self.activation,
                        &[plane_rows, shape.widths.kv_heads, shape.widths.width],
                        0x77,
                    )?)?,
                    key_rounding: shape.key_rounding,
                    form: inputs.i32s(
                        &[
                            0,
                            u64::from(shape.form.rotary_adjacent),
                            u64::from(shape.form.rotary_trailing),
                            u64::from(shape.form.output_rotary),
                            shape.stored_key(),
                        ]
                        .into_iter()
                        .chain(self.dense().then_some(shape.rounds_keys()))
                        .chain(self.dense().then_some(u64::from(shape.form.is_absorbed())))
                        .collect::<Vec<_>>(),
                        &[],
                    )?,
                    history: history.share(),
                    slab_rows: u32::try_from(view).map_err(|_| "tuning history rows exceed u32")?,
                    epsilon: self.epsilon,
                    scale,
                    gate_function,
                })
            })
            .collect()
    }
}

/// The fused entries differ in the rows they serve and in their history
/// planes; they share every other argument.
macro_rules! mix_entry {
    ($tuning:ident, $module:ident, $history:ty, $serves:expr, rows |$this:ident| $rows:expr,
     statics |$mix:ident| $statics:expr,
     |$case:ident| { $($plane:ident: $state:expr),* $(,)? }) => {
        impl $tuning {
            fn served(&self, limits: TuningLimits) -> Vec<PointShape> {
                let rows = served_row_points(limits, |rows| ($serves)(self, rows));
                // Decode rows are one sequence's (a drafter's verification)
                // or several sequences' (requests decoding together), each
                // of which reads a history of its own: a point for every
                // slot class the rows can span, beside the one sequence's.
                let across = rows
                    .iter()
                    .filter(|point| point.rows <= VERIFY_ROWS)
                    .flat_map(|point| {
                        std::iter::successors(Some(2u64), |sequences| sequences.checked_mul(2))
                            .take_while(|sequences| *sequences <= point.rows)
                            .map(|sequences| across_sequences(point, sequences))
                    })
                    .collect::<Vec<_>>();
                with_contexts(
                    limits,
                    self.0.window,
                    rows.into_iter().chain(across).collect(),
                )
            }
        }

        impl EntryTuning for $tuning {
            type Entry = $module::Entry;
            type Case = AttentionMixCase<$history>;

            fn launches(&self) -> usize {
                self.0.scopes.len()
            }

            fn bindings(&self) -> String {
                let $this = self;
                format!("{}{};codec={:?}", self.0.bindings(), $rows, self.0.codec)
            }

            fn statics(
                &self,
                _inputs: &ModelInputs<'_>,
            ) -> Result<Vec<(&'static str, u64)>, String> {
                let $mix = &self.0;
                let extra: Vec<(&'static str, u64)> = $statics.into_iter().collect();
                Ok(self.0.statics().into_iter().chain(extra).collect())
            }

            fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
                self.served(limits)
            }

            fn invocation_rows(&self, launch: &Launch, representatives: &[u64]) -> Option<u64> {
                let rows = launch.rows;
                if rows == 0 || !($serves)(self, rows) || !self.0.listing.serves(launch) {
                    return None;
                }
                if rows <= VERIFY_ROWS {
                    // Priced at the nearest representative at or below its
                    // rows, a decode launch past the largest within the row
                    // bound (6 rows under a bound of 6, represented at 1, 2
                    // and 4) keeps its demand.
                    representatives.iter().copied().filter(|at| *at <= rows).max()
                        .or_else(|| representatives.iter().copied().min())
                } else {
                    representatives.iter().copied().filter(|at| *at >= rows).min()
                }
            }

            fn served_dimensions(&self, shapes: &[PointShape]) -> Vec<BTreeMap<String, u64>> {
                let mut served = self.0.served_dimensions(shapes);
                let $mix = &self.0;
                let extra: Vec<(&'static str, u64)> = $statics.into_iter().collect();
                for dimensions in &mut served {
                    dimensions.extend(extra.iter().map(|(name, value)| (name.to_string(), *value)));
                }
                served
            }

            fn workload(&self, _cases: usize) -> seismic::WorkloadRotation {
                // Four distributions qualify the same history/layout geometry;
                // they are witnesses, not four served attention invocations.
                seismic::WorkloadRotation::Selected(vec![0])
            }

            fn rotation(
                &self,
                inputs: &mut TuningInputs<'_, '_>,
                point: &PointShape,
            ) -> Result<Vec<Self::Case>, BuildError> {
                let points = self.served(inputs.limits);
                self.0.rotation(inputs, point, points)
            }

            fn args<'a>($case: &'a mut Self::Case) -> $module::Args<'a> {
                $module::Args {
                    query: &$case.query,
                    gate: &$case.gate,
                    key: &$case.key,
                    value: &$case.value,
                    query_norm: &$case.query_norm,
                    key_norm: &$case.key_norm,
                    value_norm: &$case.value_norm,
                    sinks: &$case.sinks,
                    rotary_components: &$case.rotary_components,
                    rotary_frequencies: &$case.rotary_frequencies,
                    rotary_amplitudes: &$case.rotary_amplitudes,
                    coordinates: &$case.coordinates,
                    visible: &$case.visible,
                    fresh: &$case.fresh,
                    destinations: &$case.destinations,
                    slab_rows: $case.slab_rows,
                    $($plane: $state,)*
                    form: &$case.form,
                    epsilon: $case.epsilon,
                    scale: $case.scale,
                    gate_function: $case.gate_function,
                }
            }

            fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
                case.history.states()
            }

            generated_entry!($module, this => $module::Elements { A: this.0.activation });
        }
    };
}

/// The rows a decode entry serves: every decode row class, as one unit. Each
/// of its parameters sets a row's reduction order, and a row computes alike
/// whatever rows share its launch, so one unit owns that arithmetic for every
/// decode row and a verified row runs the arithmetic it runs alone.
fn decode_serves(rows: u64) -> bool {
    rows <= VERIFY_ROWS
}

// The prefills tune the form their binding reads: Visible (`B = 0`) or
// WorkList (`B` words).
mix_entry!(
    AttentionDecodeTuning,
    attention_decode,
    DenseMixHistory,
    |_: &AttentionDecodeTuning, rows| decode_serves(rows),
    rows |_this| "",
    statics |_mix| [],
    |case| {
        selected_rows: &case.selected_rows,
        selected: case.selected.tensor(),
        selected_slab_rows: case.selected.rows_per_slab(),
        key_block: case.key_rounding.block,
        key_element: case.key_rounding.element,
        key_scale: case.key_rounding.scale,
        key_amax_floor: case.key_rounding.amax_floor(),
        history_key: case.history.key.tensor_mut(),
        history_value: case.history.value.tensor_mut(),
    }
);
mix_entry!(
    AttentionPrefillTuning,
    attention_prefill,
    DenseMixHistory,
    |this: &AttentionPrefillTuning, rows| match this.1 {
        PrefillRows::Prefill => rows > VERIFY_ROWS,
        PrefillRows::Decode => rows <= VERIFY_ROWS,
    },
    rows |this| match this.1 {
        PrefillRows::Prefill => String::new(),
        PrefillRows::Decode => format!(",M=1..{}", VERIFY_ROWS + 1),
    },
    statics |mix| [("B", mix.words()), ("BR", mix.block_keys())],
    |case| {
        selection: &case.selection,
        block_slots: &case.block_slots,
        selected_rows: &case.selected_rows,
        selected: case.selected.tensor(),
        selected_slab_rows: case.selected.rows_per_slab(),
        key_block: case.key_rounding.block,
        key_element: case.key_rounding.element,
        key_scale: case.key_rounding.scale,
        key_amax_floor: case.key_rounding.amax_floor(),
        history_key: case.history.key.tensor_mut(),
        history_value: case.history.value.tensor_mut(),
    }
);
mix_entry!(
    AttentionDecodeK8V4Tuning,
    attention_decode_k8v4,
    AffineMixHistory,
    |_: &AttentionDecodeK8V4Tuning, rows| decode_serves(rows),
    rows |_this| "",
    statics |_mix| [],
    |case| {
        history_key_codes: case.history.key_codes.tensor_mut(),
        history_key_coefficients: case.history.key_coefficients.tensor_mut(),
        history_value_codes: case.history.value_codes.tensor_mut(),
        history_value_coefficients: case.history.value_coefficients.tensor_mut(),
    }
);
mix_entry!(
    AttentionPrefillK8V4Tuning,
    attention_prefill_k8v4,
    AffineMixHistory,
    |_: &AttentionPrefillK8V4Tuning, rows| rows > VERIFY_ROWS,
    rows |_this| "",
    statics |mix| [("B", mix.words()), ("BR", mix.block_keys())],
    |case| {
        selection: &case.selection,
        block_slots: &case.block_slots,
        history_tiles: &case.history_tiles,
        history_key_codes: case.history.key_codes.tensor_mut(),
        history_key_coefficients: case.history.key_coefficients.tensor_mut(),
        history_value_codes: case.history.value_codes.tensor_mut(),
        history_value_coefficients: case.history.value_coefficients.tensor_mut(),
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The listed and unlisted prefill kernels of the same layers each take
    /// the launches the graph gives them, so no launch is demand of both:
    /// one request's rows list the tiles they see however long its
    /// history, several requests' whose histories together exceed one
    /// request's take the kernel that lists none.
    #[test]
    fn a_prefill_launch_is_demand_of_the_kernel_that_serves_it() {
        let capacity = 65536;
        let (listed, unlisted) = (
            PrefillListing::Listed { capacity },
            PrefillListing::Unlisted { capacity },
        );
        let launch = |rows, sequences, history| Launch {
            term: 0,
            rows,
            sequences,
            history,
            projected: rows,
        };
        for launch in [
            launch(512, 1, 0),
            launch(512, 1, 65024),
            launch(16, 16, 4096),
            launch(64, 16, 4096),
        ] {
            assert!(PrefillListing::Every.serves(&launch));
            assert_ne!(listed.serves(&launch), unlisted.serves(&launch));
        }
        assert!(listed.serves(&launch(512, 1, 65024)));
        assert!(unlisted.serves(&launch(16, 16, 8192)));
    }

    /// A decode unit that serves several rows starts, beside its single-row
    /// seeds, from the first legal scalar query-group split; one bounded to
    /// a single row does not.
    /// A G4/W256 K8/V4 mix of the decode entries.
    fn decode_mix() -> AttentionMix {
        AttentionMix {
            codec: magnitude_state::KvCodec::AffineK8V4.spec(seismic::DType::BF16, 256, 256),
            listing: PrefillListing::Every,
            activation: Element::bf16(),
            shape: AttentionShape {
                form: crate::planning::AttentionForm {
                    gate: crate::planning::GateSpec::Interleaved,
                    key_value: crate::planning::KeyValueSpec::Fresh {
                        key: (),
                        value: crate::planning::ValueSpec::Projected(()),
                    },
                    query_norm: true,
                    key_norm: true,
                    value_norm: false,
                    sinks: false,
                    rotary_adjacent: false,
                    rotary_trailing: false,
                    output_rotary: false,
                    absorbed: None,
                },
                key_rounding: crate::RoundingArgs::NONE,
                widths: crate::planning::AttentionWidths {
                    hidden: 2560,
                    kv_heads: 4,
                    group: 4,
                    rotary_pairs: 64,
                    width: 256,
                    picks: 0,
                },
            },
            scopes: Vec::new(),
            epsilon: 1.0e-5,
            window: None,
            selection: None,
            picked: None,
        }
    }

    const DECODE_LIMITS: TuningLimits = TuningLimits {
        max_rows: 8,
        max_pass_rows: 8,
        max_projected_rows: 8,
        context_tokens: 256,
        image_cells: 0,
    };

    /// The rows each logical launch of `rows` rows is priced at in `tuning`
    /// (none where its kernel does not serve it).
    fn represented<T: EntryTuning>(tuning: &T, rows: u64) -> Option<u64> {
        let representatives = tuning
            .points(DECODE_LIMITS)
            .iter()
            .map(|point| point.rows)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        tuning.invocation_rows(
            &Launch {
                term: 0,
                rows,
                sequences: 1,
                history: 0,
                projected: rows,
            },
            &representatives,
        )
    }

    /// Each decode entry is one tuning unit per shape over every decode row
    /// class: its points span them all, its identity names no row band, and
    /// every decode launch, of one row or of a draft block, is demand of
    /// that unit. The prefill unit over the decode rows (where the decode
    /// entry holds no form) is likewise one unit over all of them.
    #[test]
    fn a_decode_entry_is_one_unit_over_every_decode_row() {
        let rows = |points: Vec<PointShape>| {
            points
                .iter()
                .map(|point| point.rows)
                .collect::<std::collections::BTreeSet<_>>()
        };
        let affine = AttentionDecodeK8V4Tuning(decode_mix());
        let dense = AttentionDecodeTuning(decode_mix());
        assert_eq!(rows(affine.points(DECODE_LIMITS)), [1, 2, 4, 8].into());
        assert_eq!(rows(dense.points(DECODE_LIMITS)), [1, 2, 4, 8].into());
        assert_eq!(affine.bindings(), format!("A=bf16;codec={:?}", affine.0.codec));
        assert_eq!(dense.bindings(), affine.bindings());
        let prefill = AttentionPrefillTuning(decode_mix(), PrefillRows::Decode);
        assert_eq!(rows(prefill.points(DECODE_LIMITS)), [1, 2, 4, 8].into());
        assert_eq!(prefill.bindings(), format!("A=bf16,M=1..9;codec={:?}", prefill.0.codec));
        let visible = AttentionPrefillTuning(decode_mix(), PrefillRows::Prefill);
        for logical in 1..=VERIFY_ROWS {
            for unit in [
                represented(&affine, logical),
                represented(&dense, logical),
                represented(&prefill, logical),
            ] {
                let at = unit.unwrap_or_else(|| panic!("{logical} rows have no unit"));
                assert!(at <= logical, "{logical} rows priced at {at}");
            }
            assert_eq!(represented(&visible, logical), None);
        }
        assert_eq!(represented(&affine, VERIFY_ROWS + 1), None);
        assert_eq!(represented(&prefill, VERIFY_ROWS + 1), None);
    }

    /// A drafting load's demand weighs every verify width beside the one-row
    /// step, so timings that pull one row and several to different
    /// partition counts (PARTS 32 faster at one row, PARTS 16 at two to
    /// eight) settle on one: the unit's points carry the load's weights
    /// (`apply_invocation_counts`, from the graph's drafted rounds), and one
    /// cost over all of them ranks each configuration. Weighing the one-row
    /// point alone would pick the other.
    #[test]
    fn a_drafting_loads_weights_settle_one_choice_over_every_decode_row() {
        let unit = AttentionDecodeK8V4Tuning(decode_mix());
        let method = crate::PlannedMethod::Mtp {
            greedy_proposals: (VERIFY_ROWS - 1) as u8,
            sampled_proposals: (VERIFY_ROWS - 1) as u8,
            widths: crate::DraftWidths::Adaptive,
        };
        let terms = super::super::workload::terms(DECODE_LIMITS, &method.draft_widths(), false);
        let graph = crate::programs::invocation_demand::GraphInvocationDemand::new(
            method,
            terms.clone(),
            None,
            DECODE_LIMITS.max_rows,
            DECODE_LIMITS.max_pass_rows,
        );
        let mut points = unit.points(DECODE_LIMITS);
        let mut representatives = points.iter().map(|point| point.rows).collect::<Vec<_>>();
        representatives.sort_unstable();
        representatives.dedup();
        let calls = graph
            .target
            .represented(|launch| unit.invocation_rows(launch, &representatives));
        let mut objective = super::super::Objective::new(terms);
        objective.measured = vec![1.; objective.terms.len()];
        super::super::apply_invocation_counts(&mut points, &calls, &objective);
        let weight_at = |rows: u64| {
            points
                .iter()
                .filter(|point| point.rows == rows)
                .map(|point| point.weight)
                .sum::<f64>()
        };
        for rows in [1, 2, 4, 8] {
            assert!(weight_at(rows) > 0., "no demand at {rows} rows");
        }
        let seconds = |parts: u64, rows: u64| match (parts, rows) {
            (32, 1) => 1.0,
            (32, _) => 3.0,
            _ => 2.0,
        };
        let cost = |parts: u64, demand: &[f64]| {
            let key = |parts: u64| seismic::LaunchKey {
                launches: vec![0],
                values: [("PARTS".to_owned(), parts)].into(),
            };
            let measured = |of: u64| {
                points
                    .iter()
                    .map(|point| seconds(of, point.rows))
                    .collect::<Vec<_>>()
            };
            seismic::Cost::relative(
                points.iter().map(|_| key(parts)).collect(),
                demand,
                &measured(parts),
                &measured(16),
            )
        };
        let all = points.iter().map(|point| point.weight).collect::<Vec<_>>();
        let one_row = points
            .iter()
            .map(|point| if point.rows == 1 { point.weight } else { 0. })
            .collect::<Vec<_>>();
        assert!(cost(32, &one_row).improves_on(&cost(16, &one_row)));
        assert!(!cost(32, &all).improves_on(&cost(16, &all)));
    }

    /// A case's picks name rows its index-list source holds at the position
    /// the point stands for, in a plane (`TS`) of those rows, as a served
    /// step's do: the longest history of a window shorter than the context
    /// stands for every longer prompt, whose pooled rows fill every pick.
    #[test]
    fn a_cases_picks_read_the_rows_its_source_holds() {
        let picks = 512;
        let picked = PickedRows {
            rate: 4,
            context: 32768,
        };
        let mix = AttentionMix {
            codec: magnitude_state::KvCodec::Dense.spec(seismic::DType::BF16, 512, 512),
            listing: PrefillListing::Every,
            activation: Element::bf16(),
            shape: AttentionShape {
                form: crate::planning::AttentionForm {
                    gate: crate::planning::GateSpec::None,
                    key_value: crate::planning::KeyValueSpec::Fresh {
                        key: (),
                        value: crate::planning::ValueSpec::Projected(()),
                    },
                    query_norm: false,
                    key_norm: true,
                    value_norm: false,
                    sinks: false,
                    rotary_adjacent: false,
                    rotary_trailing: false,
                    output_rotary: false,
                    absorbed: None,
                },
                key_rounding: crate::RoundingArgs::NONE,
                widths: crate::planning::AttentionWidths {
                    hidden: 4096,
                    kv_heads: 1,
                    group: 64,
                    rotary_pairs: 32,
                    width: 512,
                    picks,
                },
            },
            scopes: Vec::new(),
            epsilon: 1.0e-5,
            window: Some(128),
            selection: None,
            picked: Some(picked),
        };
        let tuning = AttentionDecodeTuning(mix.clone());
        let limits = TuningLimits {
            max_rows: 512,
            max_pass_rows: 512,
            max_projected_rows: 512,
            context_tokens: picked.context,
            image_cells: 0,
        };
        let points = tuning.points(limits);
        let served = tuning.served_dimensions(&points);
        let point = |rows: u64, sequences: u64, context: u64| {
            points
                .iter()
                .find(|point| {
                    (point.rows, point.sequences, point.context) == (rows, sequences, Some(context))
                })
                .unwrap_or_else(|| panic!("no point m{rows}s{sequences}-c{context}"))
        };
        let plane_rows = |point: &PointShape| {
            let rows = served
                .iter()
                .filter(|dimensions| {
                    dimensions["M"] == point.rows
                        && dimensions["T"] == point.context.unwrap() + point.rows
                })
                .map(|dimensions| dimensions["TS"])
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(rows.len(), 1, "{}: {rows:?}", point.label);
            rows.into_iter().next().unwrap()
        };
        // The window's longest history stands for a step at the context
        // limit, whose source holds every complete group.
        let long = point(1, 1, 128);
        assert_eq!(plane_rows(long), 32768 / 4);
        let one = mix.picks(long, &[[128, 128, 128, 0]], &[0]);
        assert_eq!(one.len(), picks as usize);
        assert!(one.iter().all(|row| (0..32768 / 4).contains(row)), "{one:?}");
        assert_eq!(
            one.iter().collect::<std::collections::BTreeSet<_>>().len(),
            picks as usize
        );
        // Sequences decoding together each read rows of their own.
        let pair = point(2, 2, 128);
        let two = mix.picks(pair, &[[128, 128, 128, 0]; 2], &[0, 1]);
        assert_eq!(
            two.iter().collect::<std::collections::BTreeSet<_>>().len(),
            2 * picks as usize
        );
        // A step that has completed no group picks none, of no rows.
        let fresh = point(1, 1, 0);
        assert_eq!(plane_rows(fresh), 0);
        assert!(mix.picks(fresh, &[[0, 0, 0, 0]], &[0]).iter().all(|row| *row == -1));
        // A step holding fewer groups than picks reads each of them.
        let short = PointShape {
            context: Some(64),
            ..fresh.clone()
        };
        let held = mix.picks(&short, &[[64, 64, 64, 0]], &[0]);
        assert_eq!(&held[..16], (0..16).collect::<Vec<_>>());
        assert!(held[16..].iter().all(|row| *row == -1));
    }

    /// A tuning point's rows see one history span; the unit names each of
    /// its points' calls at every span class a store seals graphs for, so a
    /// candidate is checked at the largest before it is chosen.
    #[test]
    fn served_dimensions_cover_every_span_class_of_every_point() {
        let mix = AttentionMix {
            codec: magnitude_state::KvCodec::AffineK8V4.spec(seismic::DType::BF16, 128, 128),
            listing: PrefillListing::Listed { capacity: 65536 },
            activation: Element::bf16(),
            shape: AttentionShape {
                form: crate::planning::AttentionForm {
                    gate: crate::planning::GateSpec::None,
                    key_value: crate::planning::KeyValueSpec::Fresh {
                        key: (),
                        value: crate::planning::ValueSpec::Projected(()),
                    },
                    query_norm: true,
                    key_norm: true,
                    value_norm: false,
                    sinks: false,
                    rotary_adjacent: false,
                    rotary_trailing: false,
                    output_rotary: false,
                    absorbed: None,
                },
                key_rounding: crate::RoundingArgs::NONE,
                widths: crate::planning::AttentionWidths {
                    hidden: 2048,
                    kv_heads: 2,
                    group: 16,
                    rotary_pairs: 32,
                    width: 128,
                    picks: 0,
                },
            },
            scopes: Vec::new(),
            epsilon: 1.0e-5,
            window: None,
            selection: None,
            picked: None,
        };
        let tuning = AttentionPrefillK8V4Tuning(mix);
        let limits = TuningLimits {
            max_rows: 512,
            max_pass_rows: 512,
            max_projected_rows: 512,
            context_tokens: 65536,
            image_cells: 0,
        };
        let points = tuning.points(limits);
        let served = tuning.served_dimensions(&points);
        let classes = [1, 2, 4, 8, 16, 32, 64];
        assert_eq!(
            magnitude_state::MAX_HISTORY_SPANS.next_power_of_two(),
            *classes.last().unwrap() as usize
        );
        assert_eq!(served.len(), points.len() * classes.len());
        let planes = crate::operators::attention::graph::affine_dimensions(tuning.0.codec).unwrap();
        for dimensions in &served {
            for (name, value) in &planes {
                assert_eq!(dimensions.get(*name), Some(value), "codec plane {name}");
            }
        }
        for point in &points {
            let context = point.context.unwrap();
            for segments in classes {
                assert!(
                    served.iter().any(|dimensions| {
                        dimensions["M"] == point.rows
                            && dimensions["T"] == context + point.rows
                            && dimensions["R"] == segments
                            && dimensions["L"] == 1
                            && dimensions["HT"] == context.div_ceil(256).max(1)
                    }),
                    "{} at {segments} spans",
                    point.label
                );
            }
        }
    }

    #[test]
    fn worklist_selection_bitmap_matches_unit_reads_density() {
        let mut mix = AttentionMix {
            codec: magnitude_state::KvCodec::AffineK8V4.spec(seismic::DType::BF16, 128, 128),
            listing: PrefillListing::Every,
            activation: Element::bf16(),
            shape: AttentionShape {
                form: crate::planning::AttentionForm {
                    gate: crate::planning::GateSpec::None,
                    key_value: crate::planning::KeyValueSpec::Fresh {
                        key: (),
                        value: crate::planning::ValueSpec::Projected(()),
                    },
                    query_norm: true,
                    key_norm: true,
                    value_norm: false,
                    sinks: false,
                    rotary_adjacent: false,
                    rotary_trailing: false,
                    output_rotary: false,
                    absorbed: None,
                },
                key_rounding: crate::RoundingArgs::NONE,
                widths: crate::planning::AttentionWidths {
                    hidden: 2048,
                    kv_heads: 2,
                    group: 16,
                    rotary_pairs: 32,
                    width: 128,
                    picks: 0,
                },
            },
            scopes: Vec::new(),
            epsilon: 1.0e-5,
            window: None,
            selection: None,
            picked: None,
        };
        assert!(mix.selection_bitmap(&[1024]).is_empty());

        let words = 2049u64;
        let block = 4u64;
        let top = 512u64;
        mix.selection = Some(SelectionBinding {
            shape: crate::SelectionShape { block, words },
            keys: crate::SelectionKeysBinding {
                projection: Element::bf16(),
                gate: None,
                place: false,
                norm: Element::f32(),
                token_norm: false,
            },
            indexer: crate::IndexerBinding {
                input: magnitude_family_contracts::IndexerInput::Hidden,
                heads: 4,
                width: 64,
                top,
                query_norm: 0,
                query: Element::bf16(),
                head_weights: None,
                candidates: crate::planning::CandidatesBinding::All,
                query_rounding: crate::RoundingArgs::NONE,
                key_rounding: crate::RoundingArgs::NONE,
            },
        });

        let row_keys = [
            top * block,       // dense: complete == top
            262_144,           // sparse: complete == 65_536, tail in word 2048
            262_148,           // consecutive row with same complete / top
            100_000,           // mid-context: complete == 25_000, partial tail word
        ];
        let bitmap = mix.selection_bitmap(&row_keys);
        assert_eq!(bitmap.len(), row_keys.len() * words as usize);

        let is_set = |row: usize, unit: u64| -> bool {
            (bitmap[row * words as usize + (unit / 32) as usize] >> (unit % 32)) & 1 != 0
        };
        let total_units = words * 32;

        // Dense row: every bit set.
        assert!(bitmap[..words as usize].iter().all(|&word| word == u32::MAX));

        // Sparse rows: exactly `top` bits in 0..complete, and every tail bit >= complete set.
        for (row, &keys) in row_keys.iter().enumerate().skip(1) {
            let complete = keys / block;
            let picked = (0..complete).filter(|&u| is_set(row, u)).count() as u64;
            assert_eq!(picked, top, "row {row} picked count");
            assert!(
                (complete..total_units).all(|u| is_set(row, u)),
                "row {row} tail bits"
            );
        }

        // Consecutive rows with the same complete / top select identical complete units.
        let shared = (0..row_keys[1] / block).all(|u| is_set(1, u) == is_set(2, u));
        assert!(shared, "consecutive rows in a tile should share selected blocks");
    }
}
