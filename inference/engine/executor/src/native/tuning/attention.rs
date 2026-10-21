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

use super::cases::{packing_statics, projection_shape};
use super::{
    CaseState, EntryTuning, ModelInputs, PointShape, TuningInputs, TuningLimits, across_sequences,
    history_placement, longest_history, row_points, served_row_points, with_contexts,
};
use crate::operators;
use crate::operators::attention::graph::{
    DECODE_ROWS, indexer_project_dimensions, rotary_amplitudes, rotary_components,
    rotary_frequencies, selected_segments,
};
use crate::programs::invocation_demand::Launch;
use crate::{AttentionBinding, AttentionShape, SelectionBinding};
use magnitude_family_contracts::{
    Attention, HeadHistory, HistoryDomain, KeyValue, Operator, WeightKind, WeightScope,
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
    implementation: &seismic::NativeImplementation,
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
        mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2) && mix.shape.width >= 512;
    let desired_parts = if wide_verification {
        default_parts
    } else {
        bytes_per_head.div_ceil(256 * 1024)
    };
    let mut starts = match backend {
        seismic::BackendName::Cuda
            if mix.shape.width <= 256
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
                    let tokens = (warps * 16 / mix.shape.group).max(1);
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
            if mix.shape.width <= 256
                && ((mix.shape.group <= 4
                    && mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2))
                    || mix.shape.group == 8) =>
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
                    let wide_group = mix.shape.group == 8;
                    let packed_g8 = wide_group
                        && mix.shape.width == 256
                        && history >= 16_384
                        && mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2);
                    let history_parts = history
                        .min(65_536)
                        .saturating_mul(mix.shape.width)
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
            let target_parts = (DECODE_GRID / mix.shape.kv_heads.max(1))
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
    /// The segments' packing bounds, query, gate, key, value.
    bounds: [Tensor; 4],
    /// The absent operand of every segment's scale port.
    absent: Tensor,
}

impl AttentionProjectTuning {
    /// The segments' weight kinds whose packing bounds the projection binds,
    /// in `attention_project`'s bound order; none for an absent segment (and
    /// for the query and gate of a key-value-only projection), as the block's
    /// graph binds them.
    fn bound_kinds(&self, query: WeightKind) -> [Option<WeightKind>; 4] {
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
            QW: b.query,
            GW: b.gate,
            KW: b.key,
            VW: b.value,
            A: b.activation,
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
            b.query.name(),
            b.gate.name(),
            b.key.name(),
            b.value.name(),
            b.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        let query = operators::attention::query_kind(operator(inputs, &self.scopes)?);
        if projection_shape(inputs, &self.scopes, query)? != (shape.query_rows(), shape.hidden) {
            return Err("the query projection disagrees with the binding".into());
        }
        let [_, statics @ ..] = if self.binding.key_value_only {
            shape.key_value_dimensions(0)
        } else {
            shape.project_dimensions(0)
        };
        let mut statics = statics.to_vec();
        statics.extend(operators::attention::UNSCALED_PROJECT);
        let [query, gate, key, value] = self.bound_kinds(query);
        statics.extend(packing_statics(
            inputs,
            &self.scopes,
            &[("QB", query), ("GB", gate), ("KB", key), ("VB", value)],
        )?);
        let (parts, norm) = inputs.input_form(&self.scopes, shape.hidden)?;
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
        let query_kind = operators::attention::query_kind(operator(inputs, &self.scopes)?);
        let (parts, norm) = inputs.input_form(&self.scopes, shape.hidden)?;
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
                        &[point.rows, parts * shape.hidden],
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
                    bounds: {
                        let kinds = self.bound_kinds(query_kind);
                        let mut bound = |kind: Option<WeightKind>| match kind {
                            Some(kind) => inputs.bound(scope, kind),
                            None => inputs.unit_scale(0),
                        };
                        [
                            bound(kinds[0])?,
                            bound(kinds[1])?,
                            bound(kinds[2])?,
                            bound(kinds[3])?,
                        ]
                    },
                    absent: inputs.unit_scale(0)?,
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
            query_scale: &case.absent,
            gate_scale: &case.absent,
            key_scale: &case.absent,
            value_scale: &case.absent,
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
    fn elements(&self) -> attention_project::Elements {
        let selection = self.selection;
        attention_project::Elements {
            NW: self.binding.norm,
            QW: selection.query,
            GW: selection.query,
            KW: selection.key,
            VW: selection.query,
            A: self.binding.activation,
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
        let hidden = self.binding.shape.hidden;
        let [_, statics @ ..] = indexer_project_dimensions(hidden, self.selection.shape, 0);
        if projection_shape(inputs, &self.scopes, WeightKind::IndexerQuery)?
            != (statics[1].1, hidden)
        {
            return Err("the indexer query projection disagrees with the binding".into());
        }
        let (parts, norm) = inputs.input_form(&self.scopes, hidden)?;
        let mut statics = [statics.as_slice(), &[("PT", parts), ("NM", norm)]].concat();
        statics.extend(operators::attention::UNSCALED_PROJECT);
        statics.extend(packing_statics(
            inputs,
            &self.scopes,
            &[
                ("QB", Some(WeightKind::IndexerQuery)),
                ("GB", None),
                ("KB", Some(WeightKind::IndexerKey)),
                ("VB", None),
            ],
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
        let hidden = self.binding.shape.hidden;
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
                    bounds: [
                        inputs.bound(scope, WeightKind::IndexerQuery)?,
                        inputs.unit_scale(0)?,
                        inputs.bound(scope, WeightKind::IndexerKey)?,
                        inputs.unit_scale(0)?,
                    ],
                    absent: inputs.unit_scale(0)?,
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
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct AttentionOutputCase {
    hidden: Tensor,
    inject: Tensor,
    gated: Tensor,
    output: Tensor,
    bound: Tensor,
    /// The absent operand of the output's scale port.
    absent: Tensor,
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
        let heads = shape.kv_heads * shape.group;
        if projection_shape(inputs, &self.scopes, WeightKind::AttentionOutput)?
            != (shape.hidden, heads * shape.width)
        {
            return Err("the attention output projection disagrees with the binding".into());
        }
        let mut statics = vec![
            ("D", shape.hidden),
            ("Q", heads),
            ("W", shape.width),
            operators::attention::UNSCALED_OUTPUT,
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
                        &[point.rows, streams * shape.hidden],
                        seed + 1,
                    )?,
                    inject: inputs.activation(Element::f32(), &[point.rows, inject], seed + 3)?,
                    gated: inputs.activation(
                        self.activation,
                        &[point.rows, shape.kv_heads * shape.group, shape.width],
                        seed + 2,
                    )?,
                    output: inputs.weight(scope, WeightKind::AttentionOutput)?,
                    bound: inputs.bound(scope, WeightKind::AttentionOutput)?,
                    absent: inputs.unit_scale(0)?,
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
            output_scale: &case.absent,
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
    /// The row classes this variant serves. Distinct ranges have distinct
    /// tuning identities and are validated on their own rows.
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
        let domain = match &operator(inputs, &[*scope])?.key_value {
            KeyValue::Owned { domain, .. } => *domain,
            KeyValue::Shared { source } => {
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
                        .history_domain(*source)
                        .ok_or_else(|| format!("{scope:?} reads no target history"))?
                } else {
                    let owner = match scope {
                        WeightScope::TargetSublayer(_) => WeightScope::TargetSublayer(*source),
                        WeightScope::HeadSublayer(_) => WeightScope::HeadSublayer(*source),
                        WeightScope::DraftSublayer(_) => WeightScope::DraftSublayer(*source),
                        other => return Err(format!("{other:?} names no sublayer")),
                    };
                    match &operator(inputs, &[owner])?.key_value {
                        KeyValue::Owned { domain, .. } => *domain,
                        KeyValue::Shared { .. } => {
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

/// Dense key and value planes `[T, KV, W]` in the activation dtype.
pub(crate) struct DenseMixHistory {
    key: CaseState,
    value: CaseState,
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
            let extents = [rows, shape.kv_heads, shape.width];
            let plane = inputs.shared(
                format!("{name}-{}-{extents:?}", mix.activation.name()),
                |inputs| inputs.activation(mix.activation, &extents, seed),
            )?;
            inputs.slab_state(plane, view, appended.clone())
        };
        Ok(Self {
            key: plane("history_key", 0x6b)?,
            value: plane("history_value", 0x76)?,
        })
    }

    fn share(&self) -> Self {
        Self {
            key: self.key.share(),
            value: self.value.share(),
        }
    }

    fn states(&self) -> Vec<(&'static str, &CaseState)> {
        vec![("history_key", &self.key), ("history_value", &self.value)]
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
                        super::f16_bits(magnitude / levels as f32).to_le_bytes(),
                        super::f16_bits(offset).to_le_bytes(),
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
                        shape.kv_heads, shape.width, mix.codec
                    ),
                    |inputs| build(inputs),
                )?;
                inputs.slab_state(plane, view, appended.clone())
            };
        let geometry = mix.codec.affine_geometry().map_err(|e| e.to_string())?;
        let pairs = [rows, shape.kv_heads, geometry.key_coefficients as u64];
        let value_pairs = [rows, shape.kv_heads, geometry.value_coefficients as u64];
        Ok(Self {
            key_codes: plane("history_key_codes", &|inputs| {
                Self::codes(
                    inputs,
                    &[rows, shape.kv_heads, geometry.key_codes as u64],
                    0x6b,
                )
            })?,
            key_coefficients: plane("history_key_coefficients", &|inputs| {
                Self::coefficients(inputs, &pairs, (1 << geometry.key_bits) - 1)
            })?,
            value_codes: plane("history_value_codes", &|inputs| {
                Self::codes(
                    inputs,
                    &[rows, shape.kv_heads, geometry.value_codes as u64],
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
        self.shape
            .mix_statics()
            .into_iter()
            .chain([("L", u64::from(self.listing.lists()))])
            .chain(
                if self.codec.encoding() == magnitude_state::HistoryEncoding::Affine {
                    crate::operators::attention::graph::affine_dimensions(self.codec)
                        .expect("validated affine descriptor")
                } else {
                    Vec::new()
                },
            )
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
        let placement = history_placement(inputs, H::CODEC, point)?;
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
        let (heads, width) = (shape.heads(), shape.width);
        let fresh_rows = [shape.fresh, rows, shape.kv_heads * width];
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 4 * index as u64;
                // Head norms are the layers' own weights, or none.
                let mut head_norm = |kind| {
                    if shape.head_norm > 0 {
                        inputs
                            .weight(scope, kind)?
                            .reshape(&[1, width])
                            .map_err(BuildError::from)
                    } else {
                        inputs.f32s(&[0, width], &[])
                    }
                };
                let query_norm = head_norm(WeightKind::QueryNorm)?;
                // A Shared layer's unread key norm port takes its query norm
                // (`graph::attention`).
                let key_norm = if shape.fresh == 0 {
                    query_norm.clone()
                } else {
                    head_norm(WeightKind::KeyNorm)?
                };
                // Sinks are the layers' own logits, or none.
                let sinks = if shape.sinks > 0 {
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
                        &[shape.value_norm, width],
                        &vec![1.0; (shape.value_norm * width) as usize],
                    )?,
                    sinks,
                    query: inputs.activation(
                        self.activation,
                        &[rows, heads, width + shape.interleaved_gate],
                        seed + 1,
                    )?,
                    gate: inputs.activation(
                        self.activation,
                        &[rows, heads, shape.separate_gate],
                        seed + 2,
                    )?,
                    key: inputs.activation(self.activation, &fresh_rows, seed + 3)?,
                    value: inputs.activation(self.activation, &fresh_rows, seed + 4)?,
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
                    ($serves)(rows)
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
                if rows == 0 || !($serves)(rows)
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
                implementation: &seismic::NativeImplementation,
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
    |rows| rows <= DECODE_ROWS,
    statics |_mix| [],
    |case| {
        history_key: case.history.key.tensor_mut(),
        history_value: case.history.value.tensor_mut(),
    },
    search_starts decode_starts
);
mix_entry!(
    AttentionPrefillTuning,
    attention_prefill,
    DenseMixHistory,
    |rows| rows > DECODE_ROWS,
    statics |mix| [("B", mix.words()), ("BR", mix.block_keys())],
    |case| {
        selection: &case.selection,
        block_slots: &case.block_slots,
        history_key: case.history.key.tensor_mut(),
        history_value: case.history.value.tensor_mut(),
    }
);
mix_entry!(
    AttentionDecodeK8V4Tuning,
    attention_decode_k8v4,
    AffineMixHistory,
    |rows| rows <= DECODE_ROWS,
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
    |rows| rows > DECODE_ROWS,
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
        for implementation in [
            seismic::generated::native_implementation_for_backend::<attention_decode::Entry>(
                seismic::BackendName::Metal,
            )
            .unwrap()
            .unwrap(),
            seismic::generated::native_implementation_for_backend::<attention_decode_k8v4::Entry>(
                seismic::BackendName::Metal,
            )
            .unwrap()
            .unwrap(),
        ] {
            for group in [1, 2, 3, 4, 8] {
                let mut mix = AttentionMix {
                    codec: magnitude_state::KvCodec::AffineK8V4.spec(
                        seismic::DType::BF16,
                        256,
                        256,
                    ),
                    listing: PrefillListing::Every,
                    activation: Element::bf16(),
                    shape: AttentionShape {
                        hidden: 2560,
                        kv_heads: 4,
                        group,
                        rotary_pairs: 32,
                        width: 256,
                        interleaved_gate: 256,
                        separate_gate: 0,
                        fresh: 1,
                        head_norm: 1,
                        value_norm: 0,
                        projected_value: true,
                        sinks: 0,
                    },
                    scopes: Vec::new(),
                    epsilon: 1.0e-5,
                    decode_rows: Some(2..DECODE_ROWS + 1),
                    window: None,
                    selection: None,
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
        .unwrap();
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
                    hidden: 1536,
                    kv_heads,
                    group,
                    rotary_pairs: 64,
                    width,
                    interleaved_gate: 0,
                    separate_gate: 0,
                    fresh: 1,
                    head_norm: 1,
                    value_norm: 1,
                    projected_value: true,
                    sinks: 0,
                },
                scopes: Vec::new(),
                epsilon: 1.0e-6,
                decode_rows: Some(rows),
                window: None,
                selection: None,
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
            .unwrap(),
            seismic::generated::native_implementation_for_backend::<attention_decode_k8v4::Entry>(
                BackendName::Cuda,
            )
            .unwrap()
            .unwrap(),
        ];
        for implementation in implementations {
            for (group, width, max_rows, expected_warps) in [
                (2, 256, 8, 1),
                (6, 256, 8, 4),
                (8, 128, 8, 4),
                (16, 256, 8, 8),
                (6, 256, 4, 2),
            ] {
                let mut mix = AttentionMix {
                    codec: magnitude_state::KvCodec::AffineK8V4.spec(
                        seismic::DType::BF16,
                        width as usize,
                        width as usize,
                    ),
                    listing: PrefillListing::Every,
                    activation: Element::bf16(),
                    shape: AttentionShape {
                        hidden: 2048,
                        kv_heads: 4,
                        group,
                        rotary_pairs: 32,
                        width,
                        interleaved_gate: 0,
                        separate_gate: 0,
                        fresh: 1,
                        head_norm: 1,
                        value_norm: 0,
                        projected_value: true,
                        sinks: 0,
                    },
                    scopes: Vec::new(),
                    epsilon: 1.0e-5,
                    decode_rows: Some(2..DECODE_ROWS + 1),
                    window: None,
                    selection: None,
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
                // The default is the vector form, except on the affine entry
                // where a lane of the whole group would hold more than 64
                // elements and the group splits further.
                let affine = implementation.statics.iter().any(|name| name == "KC");
                let default = implementation.default_specialization(&statics).unwrap();
                assert_eq!(
                    default.param("MATRIX"),
                    Some(u64::from(affine && group * width / 32 > 64))
                );

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
                mix.shape.width = 512;
                mix.codec.key_width = 512;
                mix.codec.value_width = 512;
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
                hidden: 2560,
                kv_heads: 4,
                group: 4,
                rotary_pairs: 64,
                width: 256,
                interleaved_gate: 256,
                separate_gate: 0,
                fresh: 1,
                head_norm: 1,
                value_norm: 0,
                projected_value: true,
                sinks: 0,
            },
            scopes: Vec::new(),
            epsilon: 1.0e-5,
            decode_rows: Some(1..2),
            window: None,
            selection: None,
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
                hidden: 2048,
                kv_heads: 2,
                group: 16,
                rotary_pairs: 32,
                width: 128,
                interleaved_gate: 0,
                separate_gate: 0,
                fresh: 1,
                head_norm: 1,
                value_norm: 0,
                projected_value: true,
                sinks: 0,
            },
            scopes: Vec::new(),
            epsilon: 1.0e-5,
            decode_rows: None,
            window: None,
            selection: None,
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
