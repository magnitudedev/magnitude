//! Tuning cases of the attention block: the normed segmented projection, the
//! fused attention entries of each history codec (`attention_decode` and
//! `attention_decode_k8v4` up to [`DECODE_ROWS`] rows, `attention_prefill` and
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
    CaseState, EntryTuning, ModelInputs, PointShape, TuningInputs, TuningLimits, across_sequences,
    history_placement, longest_history, row_points, served_row_points, with_contexts,
};
use crate::operators;
use crate::operators::attention::graph::{
    DECODE_ROWS, indexer_project_dimensions, query_projection_dimensions,
    rotary_amplitudes, rotary_components, rotary_frequencies, selected_segments,
};
use crate::programs::invocation_demand::Launch;
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

/// Start CUDA verification with a query tile sized for its served rows;
/// start the Vulkan key-parallel and Metal matrix forms at a bounded
/// encoded-history slice of the longest history the case's layers keep
/// (a window layer never holds the whole context), and every other Metal
/// matrix form at the partitions that fill the device for its kv heads
/// (one kv head of a long history needs many). Metal verification also
/// starts from the first legal scalar query-group split. Starts expose useful
/// candidates early; every admissible configuration remains searchable and
/// is judged on the case's served points.
/// Threadgroups a Metal decode-attention grid needs to keep an Apple GPU
/// busy over a long history (`decode_starts`).
const DECODE_GRID: u64 = 128;

fn decode_starts(
    mix: &AttentionMix,
    backend: seismic::BackendName,
    implementation: &seismic::OnDevice,
    statics: &seismic::NativeSpecialization,
    limits: TuningLimits,
) -> Vec<seismic::ParameterValues> {
    let Some(defaults) = implementation.default_specialization(statics).ok() else {
        return Vec::new();
    };
    let default_parts = defaults.param("PARTS").unwrap_or(1);
    let history = longest_history(limits, mix.window);
    // Physical history traffic follows the selected load-owned descriptor.
    let bytes_per_head = history.saturating_mul(
        mix.codec
            .bytes_per_head()
            .expect("validated history descriptor") as u64,
    );
    // Wide heads run the keywise walk in one subgroup. Across multiple
    // verification rows, extra partitions multiply its merge and dispatch
    // work; start at the default partition count and let the search compare
    // wider choices on the rows this variant actually serves.
    let wide_verification =
        mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2) && mix.shape.widths.width >= 512;
    let desired_parts = if wide_verification {
        default_parts
    } else {
        bytes_per_head.div_ceil(256 * 1024)
    };
    let mut starts = match backend {
        seismic::BackendName::Cuda
            if mix.shape.widths.width <= 256
                && mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2)
                && limits.max_rows >= 2 =>
        {
            let rows = mix.decode_rows.as_ref().unwrap();
            let widest = (rows.end - 1).min(DECODE_ROWS).min(limits.max_rows);
            nearest(
                implementation,
                statics,
                |choice| choice.param("MATRIX") == Some(1) && choice.param("QREG") == Some(1),
                |choice| {
                    let warps = choice.param("WARPS").unwrap_or(1);
                    let tokens = (warps * 16 / mix.shape.widths.group).max(1);
                    // Share history across the widest served verification
                    // tile with the fewest row warps. Register-held queries
                    // need room per thread, so start at the fewest column
                    // groups and operand stages. The declared domain and
                    // normal served-resource checks remain authoritative.
                    (
                        widest.div_ceil(tokens),
                        warps,
                        choice.param("COLUMNS").unwrap_or(1),
                        choice.param("STAGES").unwrap_or(2),
                        choice.param("PARTS").unwrap_or(1).abs_diff(default_parts),
                    )
                },
            )
        }
        seismic::BackendName::Vulkan if desired_parts > default_parts || wide_verification => {
            nearest(
                implementation,
                statics,
                |choice| choice.param("MATRIX") == Some(0) && choice.param("KEYWISE") == Some(1),
                |choice| {
                    let parts = choice.param("PARTS").unwrap_or(1);
                    let distance = defaults
                        .params()
                        .iter()
                        .filter(|(name, _)| name.as_str() != "PARTS" && name.as_str() != "KEYWISE")
                        .map(|(name, value)| choice.param(name).unwrap_or(0).abs_diff(*value))
                        .sum::<u64>();
                    (
                        parts.abs_diff(desired_parts),
                        parts < desired_parts,
                        distance,
                    )
                },
            )
        }
        seismic::BackendName::Metal
            if mix.shape.widths.width <= 256
                && ((mix.shape.widths.group <= 4
                    && mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2))
                    || mix.shape.widths.group == 8) =>
        {
            nearest(
                implementation,
                statics,
                |choice| choice.param("MATRIX") == Some(1),
                |choice| {
                    let parts = choice.param("PARTS").unwrap_or(1);
                    // The long-context G8/W256 multirow form reuses one decoded
                    // K/V tile across four verification rows. Seed it at the
                    // measured tile geometry; the tuner still compares legal
                    // configurations over all served rows and contexts.
                    let wide_group = mix.shape.widths.group == 8;
                    let packed_g8 = wide_group
                        && mix.shape.widths.width == 256
                        && history >= 16_384
                        && mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2);
                    let history_parts = history
                        .min(65_536)
                        .saturating_mul(mix.shape.widths.width)
                        .saturating_mul(7)
                        .div_ceil(4 * if wide_group { 512 * 1024 } else { 256 * 1024 });
                    // Fewer than 32 partitions underfill the G8 decode grid on
                    // the measured Apple GPUs even when history is short.
                    // The packed form runs one simdgroup per token; its K and V
                    // tiles fit 16 keys at W = 256.
                    let (tokens, target_parts, keys, simds, span) = if packed_g8 {
                        (4, history_parts.max(32).next_power_of_two(), 16, 4, 32)
                    } else if wide_group {
                        // One row: about 256 KiB of encoded history per
                        // partition. Fewer, larger partitions leave cores idle
                        // on the wider Apple GPUs (M4 Max Qwen35B 65k: P64
                        // 330 us, P128 235 us).
                        (1, history_parts.saturating_mul(2).max(32), 8, 2, 128)
                    } else {
                        (4, history_parts, 8, 4, 128)
                    };
                    (
                        choice.param("TOKENS").unwrap_or(1).abs_diff(tokens),
                        parts.abs_diff(target_parts),
                        choice.param("KEYS").unwrap_or(16).abs_diff(keys),
                        choice.param("SIMDS").unwrap_or(4).abs_diff(simds),
                        choice.param("SPAN").unwrap_or(128).abs_diff(span),
                    )
                },
            )
        }
        // Every other Metal matrix form walks a partition of its kv head's
        // history per threadgroup, in a grid of one row of partitions per kv
        // head, so the default partition count leaves the device idle over
        // long histories when there are few kv heads (Gemma 4 full layers:
        // one kv head of 512 columns, E2B 64K, four verification rows: P16
        // 2.9 ms, P128 1.8 ms). Seed the partitions that give the grid about
        // DECODE_GRID threadgroups, no partition holding under 64 KiB of
        // encoded history, rows of one token each over eight simdgroups.
        seismic::BackendName::Metal => {
            let target_parts = (DECODE_GRID / mix.shape.widths.kv_heads.max(1))
                .min(bytes_per_head.div_ceil(64 * 1024))
                .max(1)
                .next_power_of_two();
            nearest(
                implementation,
                statics,
                |choice| choice.param("MATRIX") == Some(1),
                |choice| {
                    (
                        choice.param("PARTS").unwrap_or(1).abs_diff(target_parts),
                        choice.param("TOKENS").unwrap_or(1).abs_diff(1),
                        choice.param("SIMDS").unwrap_or(4).abs_diff(8),
                        choice.param("KEYS").unwrap_or(16).abs_diff(16),
                        choice.param("SPAN").unwrap_or(32).abs_diff(32),
                    )
                },
            )
        }
        _ => Vec::new(),
    };
    // Verification can exhaust its short search after the default and matrix
    // starts. Also measure the first legal scalar query-group split: fewer
    // heads per subgroup reduce its live accumulators without changing forms.
    // The domain supplies the split and all other default parameters stay put;
    // this is a measured candidate, never an unconditional mapping choice.
    if backend == seismic::BackendName::Metal
        && mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2)
    {
        if let Some(slices) = implementation
            .params
            .iter()
            .find(|param| param.name == "SLICES")
        {
            let default_slices = defaults.param("SLICES").unwrap_or(1);
            let candidate = slices
                .values
                .iter()
                .copied()
                .filter(|value| *value > default_slices)
                .filter_map(|value| {
                    let choice = defaults.clone().with_param("SLICES", value);
                    implementation
                        .validate(&choice)
                        .ok()
                        .map(|_| (value, choice))
                })
                .min_by_key(|(value, _)| *value);
            if let Some((_, choice)) = candidate {
                starts.push(choice.params().clone());
            }
        }
    }
    starts
}

/// The parameter values of the admissible configuration `accepts` takes
/// whose `key` is least, the first of equals in declaration order: found by
/// walking the domain, never holding it.
fn nearest<K: Ord>(
    implementation: &seismic::NativeImplementation,
    statics: &seismic::NativeSpecialization,
    accepts: impl Fn(&seismic::NativeSpecialization) -> bool,
    key: impl Fn(&seismic::NativeSpecialization) -> K,
) -> Vec<seismic::ParameterValues> {
    let mut best: Option<(K, seismic::ParameterValues)> = None;
    let walked = implementation.walk_admissible(statics, |choice| {
        if accepts(choice) {
            let key = key(choice);
            if best.as_ref().is_none_or(|(least, _)| key < *least) {
                best = Some((key, choice.params().clone()));
            }
        }
        true
    });
    match walked {
        Ok(()) => best.map(|(_, values)| values).into_iter().collect(),
        Err(_) => Vec::new(),
    }
}

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
    epsilon: f32,
    /// The segments' scales, query, gate, key, value.
    scales: [Tensor; 4],
    /// The segments' packing bounds, in the same order.
    bounds: [Tensor; 4],
}

/// The unit scales of `attention_project`'s segments at `scope` (absent for
/// a segment it does not read, `None`), at their planned extents.
fn segment_scales(
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
fn segment_bounds(
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
    /// The segments' weight kinds whose scales and packing bounds the
    /// projection binds, in `attention_project`'s order; none for an absent
    /// segment (and for the query and gate of a key-value-only projection),
    /// as the block's graph binds them.
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
            GW: b.gate,
            KW: b.key,
            VW: b.value,
            A: b.activation,
            Y: b.activation,
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
            "NW={},QW={},GW={},KW={},VW={},A={}",
            b.norm.name(),
            b.query_segment().name(),
            b.gate.name(),
            b.key.name(),
            b.value.name(),
            b.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        let query = operators::attention::query_segment_kind(operator(inputs, &self.scopes)?);
        let rank = self.binding.low_rank_query.map(|low_rank| low_rank.rank);
        let [_, statics @ ..] = if self.binding.key_value_only {
            shape.key_value_dimensions(0)
        } else {
            shape.project_dimensions(0, rank)
        };
        if projection_shape(inputs, &self.scopes, query)?
            != (rank.unwrap_or(shape.query_rows()), shape.widths.hidden)
        {
            return Err("the query projection disagrees with the binding".into());
        }
        let mut statics = statics.to_vec();
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
                    scales: segment_scales(inputs, scope, self.operand_kinds(query_kind))?,
                    bounds: segment_bounds(inputs, scope, self.operand_kinds(query_kind))?,
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

/// `attention_project` as a selected read's indexer projection: the normed
/// input's query heads (`heads · width` rows) and raw key (`width` rows),
/// with no gate or value rows.
#[derive(Clone)]
pub(crate) struct IndexerProjectTuning {
    pub binding: AttentionBinding,
    pub selection: SelectionBinding,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

impl IndexerProjectTuning {
    /// The indexer's segment weights in `attention_project`'s order: a query
    /// and a key, no gate or value.
    const OPERAND_KINDS: [Option<WeightKind>; 4] = [
        Some(WeightKind::IndexerQuery),
        None,
        Some(WeightKind::IndexerKey),
        None,
    ];

    fn elements(&self) -> attention_project::Elements {
        let selection = self.selection;
        attention_project::Elements {
            NW: self.binding.norm,
            QW: selection.query,
            GW: selection.query,
            KW: selection.key,
            VW: selection.query,
            A: self.binding.activation,
            Y: self.binding.activation,
        }
    }
}

impl EntryTuning for IndexerProjectTuning {
    type Entry = attention_project::Entry;
    type Case = AttentionProjectCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "NW={},QW={},KW={},A={};indexer",
            self.binding.norm.name(),
            self.selection.query.name(),
            self.selection.key.name(),
            self.binding.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let hidden = self.binding.shape.widths.hidden;
        let [_, statics @ ..] = indexer_project_dimensions(
            hidden,
            self.selection.shape.heads,
            self.selection.shape.width,
            0,
        );
        if projection_shape(inputs, &self.scopes, WeightKind::IndexerQuery)?
            != (statics[1].1, hidden)
        {
            return Err("the indexer query projection disagrees with the binding".into());
        }
        let (parts, norm) = inputs.input_form(&self.scopes, hidden)?;
        let mut statics = [statics.as_slice(), &[("PT", parts), ("NM", norm)]].concat();
        let [query, gate, key, value] = Self::OPERAND_KINDS;
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
        let hidden = self.binding.shape.widths.hidden;
        let (parts, norm) = inputs.input_form(&self.scopes, hidden)?;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let query = inputs.weight(scope, WeightKind::IndexerQuery)?;
                let empty = query
                    .slice_leading(0, 0)
                    .map_err(|error| error.to_string())?;
                Ok(AttentionProjectCase {
                    gate: empty.clone(),
                    key: inputs.weight(scope, WeightKind::IndexerKey)?,
                    value: empty,
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, parts * hidden],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.input_norm(
                        scope,
                        WeightKind::InputNorm,
                        self.binding.norm,
                        norm,
                    )?,
                    query,
                    epsilon: self.epsilon,
                    scales: segment_scales(inputs, scope, Self::OPERAND_KINDS)?,
                    bounds: segment_bounds(inputs, scope, Self::OPERAND_KINDS)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_project::Args<'a> {
        AttentionProjectTuning::args(case)
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
            A: self.binding.activation,
            Y: self.binding.activation,
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
        let shape = self.binding.shape;
        let rank = self.low_rank.rank;
        let query = operators::attention::query_kind(operator(inputs, &self.scopes)?);
        if projection_shape(inputs, &self.scopes, query)? != (shape.query_rows(), rank) {
            return Err("the low-rank query's up projection disagrees with the binding".into());
        }
        let [_, statics @ ..] = query_projection_dimensions(0, rank, shape.query_rows());
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
                    value: empty,
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, rank],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.weight(scope, WeightKind::QueryDownNorm)?,
                    query,
                    epsilon: epsilon as f32,
                    scales: segment_scales(inputs, scope, Self::operand_kinds(query_kind))?,
                    bounds: segment_bounds(inputs, scope, Self::operand_kinds(query_kind))?,
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
    pub elements: [(&'static str, Element); 7],
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
            A: self.element("A"),
            Y: self.element("Y"),
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
                (WeightKind::PooledNorm, rows.norm.epsilon as f32)
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
                    value: empty,
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, parts * width],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.input_norm(scope, norm, norm_element, norm_extent)?,
                    query,
                    epsilon,
                    scales: segment_scales(inputs, scope, self.operand_kinds())?,
                    bounds: segment_bounds(inputs, scope, self.operand_kinds())?,
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
    /// The decode row classes this variant serves, of a decode entry or of
    /// the prefill entry where it attends them (`graph::dense_decodes`);
    /// none for every class its entry serves by default. Distinct ranges
    /// have distinct tuning identities and are validated on their own rows.
    pub decode_rows: Option<Range<u64>>,
    /// For the affine prefill: which of the layers' prefill launches the
    /// kernel serves, and whether it lists the history row tiles their rows
    /// see (the entry's `L`).
    pub listing: PrefillListing,
    /// The most history rows the layers the case binds keep
    /// ([`history_window`]): its points see no longer history.
    pub window: Option<u64>,
    /// A selected read's indexer: the prefill then reads through its
    /// WorkList bitmap (`B` words, static), tuned with every block read.
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

/// `attention_decode`, for row classes up to [`DECODE_ROWS`].
#[derive(Clone)]
pub(crate) struct AttentionDecodeTuning(pub AttentionMix);

/// `attention_prefill`, for row classes beyond [`DECODE_ROWS`].
#[derive(Clone)]
pub(crate) struct AttentionPrefillTuning(pub AttentionMix);

/// `attention_decode_k8v4`, for row classes up to [`DECODE_ROWS`].
#[derive(Clone)]
pub(crate) struct AttentionDecodeK8V4Tuning(pub AttentionMix);

/// `attention_prefill_k8v4`, for row classes beyond [`DECODE_ROWS`].
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
    /// The prefill's WorkList bitmap (`[rows, B]`, every block read; none in
    /// the Visible form) and its blocks' keys (`[0, BR]`, BR 1 in the
    /// Visible form).
    selection: Tensor,
    block_slots: Tensor,
    /// The dense entries' index-list source: each row's picks (`[rows,
    /// KS]`) of the rows its source holds, in a plane of the most any row
    /// holds (`[TS, KV, W]`), and source 0's key rounding.
    selected_rows: Tensor,
    selected: Tensor,
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
        // A selected read's decode rows also read the indexer's segments.
        let classes = std::iter::successors(Some(1u64), |segments| segments.checked_mul(2))
            .take_while(|segments| {
                *segments <= magnitude_state::MAX_HISTORY_SPANS.next_power_of_two() as u64
            })
            .map(|segments| (segments, false))
            .chain(
                self.selection
                    .map(|selection| (selected_segments(selection.shape.blocks), true)),
            );
        classes
            .flat_map(|(segments, selected)| {
                shapes.iter().filter_map(move |point| {
                    if selected && point.rows > DECODE_ROWS {
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
        match &self.decode_rows {
            Some(rows) => format!(
                "A={},M={}..{}",
                self.activation.name(),
                rows.start,
                rows.end
            ),
            None if self.listing.lists() => format!("A={},L=1", self.activation.name()),
            None if self.selection.is_some() => {
                format!(
                    "A={},B={},BR={}",
                    self.activation.name(),
                    self.words(),
                    self.block_keys()
                )
            }
            None => format!("A={}", self.activation.name()),
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
        let fresh_rows = [u64::from(shape.form.fresh), rows, shape.widths.kv_heads * width];
        let selected_rows = self.picks(
            point,
            &tables.coordinates[..rows as usize],
            &tables.row_slots[..rows as usize],
        );
        let plane_rows = self.picked_plane_rows(point);
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
                let key_norm = if !shape.form.fresh {
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
                            u64::from(shape.form.fresh && !shape.form.stored_key()),
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
                    selection: inputs.u32s(
                        &[rows, self.words()],
                        &vec![u32::MAX; (rows * self.words()) as usize],
                    )?,
                    block_slots: inputs.i32s(&[0, self.block_keys()], &[])?,
                    selected_rows: inputs.i32s(&[rows, shape.widths.picks], &selected_rows)?,
                    selected: inputs.activation(
                        self.activation,
                        &[plane_rows, shape.widths.kv_heads, shape.widths.width],
                        0x77,
                    )?,
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
    ($tuning:ident, $module:ident, $history:ty, $serves:expr, statics |$mix:ident| $statics:expr,
     |$case:ident| { $($plane:ident: $state:expr),* $(,)? }
     $(, search_starts $starts:path)?) => {
        impl $tuning {
            fn served(&self, limits: TuningLimits) -> Vec<PointShape> {
                let rows = served_row_points(limits, |rows| {
                    ($serves)(&self.0, rows)
                        && self
                            .0
                            .decode_rows
                            .as_ref()
                            .is_none_or(|range| range.contains(&rows))
                });
                // Decode rows are one sequence's (a drafter's verification)
                // or several sequences' (requests decoding together), each
                // of which reads a history of its own: a point for every
                // slot class the rows can span, beside the one sequence's.
                let across = rows
                    .iter()
                    .filter(|point| point.rows <= DECODE_ROWS)
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
                format!("{};codec={:?}", self.0.bindings(), self.0.codec)
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
                if rows == 0 || !($serves)(&self.0, rows)
                    || self.0.decode_rows.as_ref().is_some_and(|range| !range.contains(&rows))
                    || !self.0.listing.serves(launch)
                {
                    return None;
                }
                if rows <= DECODE_ROWS {
                    // A dedicated upper decode band can begin below its
                    // first representative (5..9 is represented by 8).
                    // Keep those owned calls rather than dropping their
                    // demand; the support/range guard above prevents a
                    // call from crossing into another entry's row band.
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

            $(fn search_starts(
                &self,
                device: &Device,
                implementation: &seismic::OnDevice,
                statics: &seismic::NativeSpecialization,
                limits: TuningLimits,
            ) -> Vec<seismic::ParameterValues> {
                $starts(&self.0, device.backend(), implementation, statics, limits)
            })?

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

// The prefills tune the form their binding reads: Visible (`B = 0`) or
// WorkList (`B` words).
mix_entry!(
    AttentionDecodeTuning,
    attention_decode,
    DenseMixHistory,
    |_: &AttentionMix, rows| rows <= DECODE_ROWS,
    statics |_mix| [],
    |case| {
        selected_rows: &case.selected_rows,
        selected: &case.selected,
        selected_slab_rows: 1,
        key_block: case.key_rounding.block,
        key_element: case.key_rounding.element,
        key_scale: case.key_rounding.scale,
        key_amax_floor: case.key_rounding.amax_floor(),
        history_key: case.history.key.tensor_mut(),
        history_value: case.history.value.tensor_mut(),
    },
    search_starts decode_starts
);
mix_entry!(
    AttentionPrefillTuning,
    attention_prefill,
    DenseMixHistory,
    |mix: &AttentionMix, rows| rows > DECODE_ROWS || mix.decode_rows.is_some(),
    statics |mix| [("B", mix.words()), ("BR", mix.block_keys())],
    |case| {
        selection: &case.selection,
        block_slots: &case.block_slots,
        selected_rows: &case.selected_rows,
        selected: &case.selected,
        selected_slab_rows: 1,
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
    |_: &AttentionMix, rows| rows <= DECODE_ROWS,
    statics |_mix| [],
    |case| {
        history_key_codes: case.history.key_codes.tensor_mut(),
        history_key_coefficients: case.history.key_coefficients.tensor_mut(),
        history_value_codes: case.history.value_codes.tensor_mut(),
        history_value_coefficients: case.history.value_coefficients.tensor_mut(),
    },
    search_starts decode_starts
);
mix_entry!(
    AttentionPrefillK8V4Tuning,
    attention_prefill_k8v4,
    AffineMixHistory,
    |mix: &AttentionMix, rows| rows > DECODE_ROWS || mix.decode_rows.is_some(),
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

    #[test]
    fn metal_verification_exposes_legal_scalar_group_split() {
        // Each entry's own history codec: the dense entry's dense, the
        // affine entry's K8/V4.
        for (implementation, codec) in [
            seismic::generated::native_implementation_for_backend::<attention_decode::Entry>(
                seismic::BackendName::Metal,
            )
            .unwrap()
            .unwrap()
            .device_independent()
            .expect("the declaration reads no device property")
            .owned(),
            seismic::generated::native_implementation_for_backend::<attention_decode_k8v4::Entry>(
                seismic::BackendName::Metal,
            )
            .unwrap()
            .unwrap()
            .device_independent()
            .expect("the declaration reads no device property")
            .owned(),
        ]
        .into_iter()
        .zip([magnitude_state::KvCodec::Dense, magnitude_state::KvCodec::AffineK8V4])
        {
            for group in [1, 2, 3, 4, 8] {
                let mut mix = AttentionMix {
                    codec: codec.spec(seismic::DType::BF16, 256, 256),
                    listing: PrefillListing::Every,
                    activation: Element::bf16(),
                    shape: AttentionShape {
                        form: crate::planning::AttentionForm {
                            gate: crate::planning::GateForm::Interleaved,
                            fresh: true,
                            query_norm: true,
                            key_norm: true,
                            value_norm: false,
                            values: crate::AttentionValues::Projected,
                            sinks: false,
                            rotary_adjacent: false,
                            rotary_trailing: false,
                            output_rotary: false,
                        },
                        key_rounding: crate::RoundingArgs::NONE,
                        widths: crate::planning::AttentionWidths {
                            hidden: 2560,
                            kv_heads: 4,
                            group,
                            rotary_pairs: 32,
                            width: 256,
                            picks: 0,
                        },
                    },
                    scopes: Vec::new(),
                    epsilon: 1.0e-5,
                    decode_rows: Some(2..DECODE_ROWS + 1),
                    window: None,
                    selection: None,
                    picked: None,
                };
                let limits = TuningLimits {
                    max_rows: 8,
                    max_projected_rows: 8,
                    context_tokens: 4096,
                };
                let statics = mix
                    .statics()
                    .into_iter()
                    .filter(|(name, _)| implementation.statics.iter().any(|item| item == name))
                    .fold(
                        seismic::NativeSpecialization::new(),
                        |spec, (name, value)| spec.with_static(name, value),
                    );
                let defaults = implementation.default_specialization(&statics).unwrap();
                let starts = decode_starts(
                    &mix,
                    seismic::BackendName::Metal,
                    &implementation,
                    &statics,
                    limits,
                );
                let scalar = starts
                    .iter()
                    .filter(|choice| choice.get("MATRIX") == Some(&0))
                    .collect::<Vec<_>>();
                // The start is the next split above the default's: the
                // affine entry already defaults to two slices of an even
                // group.
                let default_slices = defaults.param("SLICES").unwrap();
                let split = 2 * default_slices;
                assert_eq!(scalar.len(), usize::from(group % split == 0));
                for values in scalar {
                    let mut expected = defaults.params().clone();
                    expected.insert("SLICES".into(), split);
                    assert_eq!(*values, expected);
                    implementation
                        .validate(&defaults.clone().with_param("SLICES", split))
                        .unwrap();
                }
                mix.decode_rows = Some(1..2);
                assert!(
                    decode_starts(
                        &mix,
                        seismic::BackendName::Metal,
                        &implementation,
                        &statics,
                        limits
                    )
                    .iter()
                    .all(|choice| choice.get("MATRIX") != Some(&0))
                );
            }
        }
    }

    #[test]
    fn metal_matrix_start_fills_the_grid_for_its_kv_heads() {
        let implementation = seismic::generated::native_implementation_for_backend::<
            attention_decode_k8v4::Entry,
        >(seismic::BackendName::Metal)
        .unwrap()
        .unwrap()
            .device_independent()
            .expect("the declaration reads no device property")
            .owned();
        // Gemma 4 E2B and 12B full layers (one kv head, 512 columns), Gemma
        // 4 31B full layers (four), Qwen 3.8 27B (four kv heads of six).
        for (kv_heads, group, width, rows, expected) in [
            (1, 8, 512, 1..2, 128),
            (1, 8, 512, 2..DECODE_ROWS + 1, 128),
            (1, 16, 512, 2..DECODE_ROWS + 1, 128),
            (4, 8, 512, 2..DECODE_ROWS + 1, 32),
            (4, 6, 256, 2..DECODE_ROWS + 1, 32),
        ] {
            let mix = AttentionMix {
                codec: magnitude_state::KvCodec::AffineK8V4.spec(
                    seismic::DType::BF16,
                    width as usize,
                    width as usize,
                ),
                listing: PrefillListing::Every,
                activation: Element::bf16(),
                shape: AttentionShape {
                    form: crate::planning::AttentionForm {
                        gate: crate::planning::GateForm::None,
                        fresh: true,
                        query_norm: true,
                        key_norm: true,
                        value_norm: true,
                        values: crate::AttentionValues::Projected,
                        sinks: false,
                        rotary_adjacent: false,
                        rotary_trailing: false,
                        output_rotary: false,
                    },
                    key_rounding: crate::RoundingArgs::NONE,
                    widths: crate::planning::AttentionWidths {
                        hidden: 1536,
                        kv_heads,
                        group,
                        rotary_pairs: 64,
                        width,
                        picks: 0,
                    },
                },
                scopes: Vec::new(),
                epsilon: 1.0e-6,
                decode_rows: Some(rows),
                window: None,
                selection: None,
                picked: None,
            };
            let statics = mix
                .statics()
                .into_iter()
                .filter(|(name, _)| implementation.statics.iter().any(|item| item == name))
                .fold(
                    seismic::NativeSpecialization::new(),
                    |spec, (name, value)| spec.with_static(name, value),
                );
            let parts = |context_tokens| {
                let limits = TuningLimits {
                    max_rows: 8,
                    max_projected_rows: 8,
                    context_tokens,
                };
                let starts = decode_starts(
                    &mix,
                    seismic::BackendName::Metal,
                    &implementation,
                    &statics,
                    limits,
                );
                let matrix = starts
                    .iter()
                    .find(|choice| choice.get("MATRIX") == Some(&1))
                    .expect("a wide head has a matrix start");
                matrix["PARTS"]
            };
            assert_eq!(parts(65_536), expected);
            // A short history keeps its partitions large.
            assert!(parts(1_024) < parts(65_536));
        }
    }

    #[test]
    fn cuda_verification_start_follows_served_rows_and_declared_domain() {
        use seismic::{BackendName, NativeSpecialization};
        let implementations = [
            seismic::generated::native_implementation_for_backend::<attention_decode::Entry>(
                BackendName::Cuda,
            )
            .unwrap()
            .unwrap()
            .device_independent()
            .expect("the declaration reads no device property")
            .owned(),
            seismic::generated::native_implementation_for_backend::<attention_decode_k8v4::Entry>(
                BackendName::Cuda,
            )
            .unwrap()
            .unwrap()
            .device_independent()
            .expect("the declaration reads no device property")
            .owned(),
        ];
        // Each entry's own history codec: the dense entry's dense, the affine
        // entry's K8/V4.
        let codecs = [magnitude_state::KvCodec::Dense, magnitude_state::KvCodec::AffineK8V4];
        for (implementation, codec) in implementations.into_iter().zip(codecs) {
            // The affine entry's default SLICES: the fewest that hold a lane
            // to 32 elements, or the most the group splits into.
            for (group, width, max_rows, expected_warps, affine_slices) in [
                (2, 256, 8, 1, 1),
                (6, 256, 8, 4, 2),
                (8, 128, 8, 4, 1),
                (16, 256, 8, 8, 4),
                (6, 256, 4, 2, 2),
            ] {
                let mut mix = AttentionMix {
                    codec: codec.spec(seismic::DType::BF16, width as usize, width as usize),
                    listing: PrefillListing::Every,
                    activation: Element::bf16(),
                    shape: AttentionShape {
                        form: crate::planning::AttentionForm {
                            gate: crate::planning::GateForm::None,
                            fresh: true,
                            query_norm: true,
                            key_norm: true,
                            value_norm: false,
                            values: crate::AttentionValues::Projected,
                            sinks: false,
                            rotary_adjacent: false,
                            rotary_trailing: false,
                            output_rotary: false,
                        },
                        key_rounding: crate::RoundingArgs::NONE,
                        widths: crate::planning::AttentionWidths {
                            hidden: 2048,
                            kv_heads: 4,
                            group,
                            rotary_pairs: 32,
                            width,
                            picks: 0,
                        },
                    },
                    scopes: Vec::new(),
                    epsilon: 1.0e-5,
                    decode_rows: Some(2..DECODE_ROWS + 1),
                    window: None,
                    selection: None,
                    picked: None,
                };
                let limits = TuningLimits {
                    max_rows,
                    max_projected_rows: max_rows,
                    context_tokens: 65536,
                };
                let statics = mix
                    .statics()
                    .into_iter()
                    .filter(|(name, _)| implementation.statics.iter().any(|item| item == name))
                    .fold(NativeSpecialization::new(), |spec, (name, value)| {
                        spec.with_static(name, value)
                    });
                let start =
                    decode_starts(&mix, BackendName::Cuda, &implementation, &statics, limits);
                assert_eq!(start.len(), 1);
                let mut choice = statics.clone();
                for (name, value) in &start[0] {
                    choice = choice.with_param(name.clone(), *value);
                }
                implementation.validate(&choice).unwrap();
                assert_eq!(choice.param("QREG"), Some(1));
                assert_eq!(choice.param("MATRIX"), Some(1));
                assert_eq!(choice.param("WARPS"), Some(expected_warps));
                assert_eq!(choice.param("COLUMNS"), Some(1));
                assert_eq!(choice.param("STAGES"), Some(2));
                // The default is the vector form, the group whole except on
                // the affine entry, which splits it so a lane holds at most
                // 32 elements where it can.
                let affine = implementation.statics.iter().any(|name| name == "KC");
                let default = implementation.default_specialization(&statics).unwrap();
                assert_eq!(default.param("MATRIX"), Some(0));
                assert_eq!(default.param("SLICES"), Some(if affine { affine_slices } else { 1 }));

                mix.decode_rows = Some(1..2);
                assert!(
                    decode_starts(&mix, BackendName::Cuda, &implementation, &statics, limits)
                        .is_empty()
                );
                mix.decode_rows = None;
                assert!(
                    decode_starts(&mix, BackendName::Cuda, &implementation, &statics, limits)
                        .is_empty()
                );
                mix.decode_rows = Some(2..DECODE_ROWS + 1);
                mix.shape.widths.width = 512;
                mix.codec.key_width = 512;
                if let magnitude_state::HistoryValue::Own { width, .. } = &mut mix.codec.value {
                    *width = 512;
                }
                let wide_statics = statics.clone().with_static("S", 448);
                assert!(
                    decode_starts(
                        &mix,
                        BackendName::Cuda,
                        &implementation,
                        &wide_statics,
                        limits
                    )
                    .is_empty()
                );
            }
        }
    }

    #[test]
    fn verify_decode_has_distinct_tuning_identity_and_served_rows() {
        use crate::native::preparation::separates_packed_attention_rows;
        for group in 1..=8 {
            for width in [128, 256] {
                assert!(separates_packed_attention_rows(
                    seismic::BackendName::Metal,
                    group,
                    width,
                    8,
                ));
                // TOKENS=SIMDS=4, KEYS=16 is a legal shared tile throughout
                // this capability. The planner does not select that tuple.
                let shared_bytes = (4 * 8 * width * 2).max(16 * width * 4) + 64;
                assert!(shared_bytes <= 32768);
            }
        }
        for (backend, group, width, rows) in [
            (seismic::BackendName::Cuda, 8, 256, 8),
            (seismic::BackendName::Metal, 16, 256, 8),
            (seismic::BackendName::Metal, 8, 512, 8),
            (seismic::BackendName::Metal, 8, 256, 3),
        ] {
            assert!(!separates_packed_attention_rows(
                backend, group, width, rows
            ));
        }
        let mix = AttentionMix {
            codec: magnitude_state::KvCodec::AffineK8V4.spec(seismic::DType::BF16, 256, 256),
            listing: PrefillListing::Every,
            activation: Element::bf16(),
            shape: AttentionShape {
                form: crate::planning::AttentionForm {
                    gate: crate::planning::GateForm::Interleaved,
                    fresh: true,
                    query_norm: true,
                    key_norm: true,
                    value_norm: false,
                    values: crate::AttentionValues::Projected,
                    sinks: false,
                    rotary_adjacent: false,
                    rotary_trailing: false,
                    output_rotary: false,
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
            decode_rows: Some(1..2),
            window: None,
            selection: None,
            picked: None,
        };
        let single = AttentionDecodeK8V4Tuning(mix);
        let limits = TuningLimits {
            max_rows: 8,
            max_projected_rows: 8,
            context_tokens: 256,
        };
        // Each served row is measured at several contexts.
        assert_eq!(
            single
                .points(limits)
                .iter()
                .map(|point| point.rows)
                .collect::<std::collections::BTreeSet<_>>(),
            [1].into()
        );
        assert_eq!(
            single.bindings(),
            format!("A=bf16,M=1..2;codec={:?}", single.0.codec)
        );
        let mut mix = single.0;
        mix.decode_rows = Some(2..DECODE_ROWS + 1);
        let verify = AttentionDecodeK8V4Tuning(mix);
        assert_eq!(
            verify
                .points(limits)
                .iter()
                .map(|point| point.rows)
                .collect::<std::collections::BTreeSet<_>>(),
            [2, 4, 8].into()
        );
        assert_eq!(
            verify.bindings(),
            format!("A=bf16,M=2..9;codec={:?}", verify.0.codec)
        );

        // Dense and compressed history use disjoint decode/verification
        // row domains on every backend with split graph dispatch. A
        // multirow cache identity cannot reuse the old combined
        // result, and neither class can train on the other's row counts.
        let dense_verify = AttentionDecodeTuning(verify.0.clone());
        let mut dense_single_mix = verify.0.clone();
        dense_single_mix.decode_rows = Some(1..2);
        let dense_single = AttentionDecodeTuning(dense_single_mix);
        assert_eq!(
            dense_single.bindings(),
            format!("A=bf16,M=1..2;codec={:?}", dense_single.0.codec)
        );
        assert_eq!(dense_verify.bindings(), verify.bindings());
        assert!(
            dense_single
                .points(limits)
                .iter()
                .all(|point| point.rows == 1)
        );
        assert_eq!(dense_verify.points(limits), verify.points(limits));

        let mut classes = Vec::new();
        for range in [2..4, 4..5, 5..DECODE_ROWS + 1] {
            let mut mix = verify.0.clone();
            mix.decode_rows = Some(range.clone());
            let tuned = AttentionDecodeK8V4Tuning(mix);
            let rows = tuned
                .points(limits)
                .iter()
                .map(|point| point.rows)
                .collect::<Vec<_>>();
            assert!(rows.iter().all(|row| range.contains(row)));
            let representatives = rows
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            for logical in 1..=DECODE_ROWS {
                let launch = Launch {
                    term: 0,
                    rows: logical,
                    sequences: 1,
                    history: 0,
                };
                let represented = tuned.invocation_rows(&launch, &representatives);
                assert_eq!(represented.is_some(), range.contains(&logical));
                assert!(represented.is_none_or(|row| range.contains(&row)));
            }
            classes.extend(range);
        }
        assert_eq!(classes, (2..DECODE_ROWS + 1).collect::<Vec<_>>());
        // The base decode plus the dedicated verification domains own
        // every physical decode row exactly once. This also excludes
        // duplicate workload pricing by a combined base domain.
        let owned = std::iter::once(1).chain(classes).collect::<Vec<_>>();
        assert_eq!(owned, (1..DECODE_ROWS + 1).collect::<Vec<_>>());
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
                    gate: crate::planning::GateForm::None,
                    fresh: true,
                    query_norm: false,
                    key_norm: true,
                    value_norm: false,
                    values: crate::AttentionValues::Projected,
                    sinks: false,
                    rotary_adjacent: false,
                    rotary_trailing: false,
                    output_rotary: false,
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
            decode_rows: None,
            window: Some(128),
            selection: None,
            picked: Some(picked),
        };
        let tuning = AttentionDecodeTuning(mix.clone());
        let limits = TuningLimits {
            max_rows: 512,
            max_projected_rows: 512,
            context_tokens: picked.context,
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
                    gate: crate::planning::GateForm::None,
                    fresh: true,
                    query_norm: true,
                    key_norm: true,
                    value_norm: false,
                    values: crate::AttentionValues::Projected,
                    sinks: false,
                    rotary_adjacent: false,
                    rotary_trailing: false,
                    output_rotary: false,
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
            decode_rows: None,
            window: None,
            selection: None,
            picked: None,
        };
        let tuning = AttentionPrefillK8V4Tuning(mix);
        let limits = TuningLimits {
            max_rows: 512,
            max_projected_rows: 512,
            context_tokens: 65536,
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
}
