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

use super::cases::projection_shape;
use super::{
    row_points, served_row_points, with_contexts, CaseState, EntryTuning, PointShape, TuningInputs,
    TuningLimits,
};
use crate::operators;
use crate::operators::attention::graph::{
    affine_coefficients, rotary_amplitudes, rotary_components, rotary_frequencies, DECODE_ROWS,
};
use crate::{AttentionBinding, AttentionShape};
use magnitude_family_contracts::{Attention, Operator, WeightKind, WeightScope};
use magnitude_kernels::{
    attention_decode, attention_decode_k8v4, attention_output, attention_prefill,
    attention_prefill_k8v4, attention_project,
};
use seismic::{Device, Element, Tensor};
use std::ops::Range;

/// Start the Vulkan key-parallel form and Metal's grouped-query matrix
/// forms at a bounded encoded-history slice. A start changes only the form's
/// first measurement; every admissible configuration remains searchable and
/// is judged on the case's served points.
fn decode_starts(
    mix: &AttentionMix,
    device: &Device,
    implementation: &seismic::NativeImplementation,
    statics: &seismic::NativeSpecialization,
    limits: TuningLimits,
) -> Vec<seismic::ParameterValues> {
    let Some(defaults) = implementation.default_specialization(statics).ok() else {
        return Vec::new();
    };
    let Some(admissible) = implementation.admissible(statics).ok() else {
        return Vec::new();
    };
    let default_parts = defaults.param("PARTS").unwrap_or(1);
    // K8/V4 codes and the scale/zero pairs use 7W/4 bytes per KV head row.
    let bytes_per_head = limits
        .context_tokens
        .saturating_mul(mix.shape.width)
        .saturating_mul(7)
        / 4;
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
    match device.backend() {
        seismic::BackendName::Vulkan if desired_parts > default_parts || wide_verification => {
            admissible
                .into_iter()
                .filter(|choice| {
                    choice.param("MATRIX") == Some(0) && choice.param("KEYWISE") == Some(1)
                })
                .min_by_key(|choice| {
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
                })
                .map(|choice| choice.params().clone())
                .into_iter()
                .collect()
        }
        seismic::BackendName::Metal
            if mix.shape.width <= 256
                && ((mix.shape.group <= 4
                    && mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2))
                    || mix.shape.group == 8) =>
        {
            admissible
                .into_iter()
                .filter(|choice| choice.param("MATRIX") == Some(1))
                .min_by_key(|choice| {
                    let parts = choice.param("PARTS").unwrap_or(1);
                    // The long-context G8/W256 multirow form reuses one decoded
                    // K/V tile across four verification rows. Seed it at the
                    // measured tile geometry; the tuner still compares legal
                    // configurations over all served rows and contexts.
                    let wide_group = mix.shape.group == 8;
                    let packed_g8 = wide_group
                        && mix.shape.width == 256
                        && limits.context_tokens >= 16_384
                        && mix.decode_rows.as_ref().is_some_and(|rows| rows.start >= 2);
                    let history_parts = limits
                        .context_tokens
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
                        (1, history_parts.max(32), 8, 2, 128)
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
                })
                .map(|choice| choice.params().clone())
                .into_iter()
                .collect()
        }
        _ => Vec::new(),
    }
}

/// `attention_project`: RMS prologue, one segmented query | gate | key |
/// value projection.
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
}

impl AttentionProjectTuning {
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
    inputs: &'i TuningInputs<'_, '_>,
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

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        let query = operators::attention::query_kind(operator(inputs, &self.scopes)?);
        if projection_shape(inputs, &self.scopes, query)? != (shape.query_rows(), shape.hidden) {
            return Err("the query projection disagrees with the binding".into());
        }
        let [_, statics @ ..] = shape.project_dimensions(0);
        Ok(statics.to_vec())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let shape = self.binding.shape;
        let query_kind = operators::attention::query_kind(operator(inputs, &self.scopes)?);
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
                        query.slice_leading(0, 0).map_err(|error| error.to_string())
                    }
                };
                Ok(AttentionProjectCase {
                    gate: segment(shape.gate_rows(), WeightKind::AttentionGate)?,
                    key: segment(shape.key_rows(), WeightKind::Key)?,
                    value: segment(shape.value_rows(), WeightKind::Value)?,
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, shape.hidden],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.weight(scope, WeightKind::InputNorm)?,
                    query,
                    epsilon: self.epsilon,
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
        }
    }

    generated_entry!(attention_project, this => this.elements());
}

/// `attention_output`: output projection plus residual.
pub(crate) struct AttentionOutputTuning {
    pub output: Element,
    pub activation: Element,
    pub shape: AttentionShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct AttentionOutputCase {
    hidden: Tensor,
    gated: Tensor,
    output: Tensor,
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

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let heads = shape.kv_heads * shape.group;
        if projection_shape(inputs, &self.scopes, WeightKind::AttentionOutput)?
            != (shape.hidden, heads * shape.width)
        {
            return Err("the attention output projection disagrees with the binding".into());
        }
        Ok(vec![("D", shape.hidden), ("Q", heads), ("W", shape.width)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let shape = self.shape;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 2 * index as u64;
                Ok(AttentionOutputCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, shape.hidden],
                        seed + 1,
                    )?,
                    gated: inputs.activation(
                        self.activation,
                        &[point.rows, shape.kv_heads * shape.group, shape.width],
                        seed + 2,
                    )?,
                    output: inputs.weight(scope, WeightKind::AttentionOutput)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_output::Args<'a> {
        attention_output::Args {
            hidden: &case.hidden,
            gated: &case.gated,
            output_weight: &case.output,
        }
    }

    generated_entry!(attention_output, this => this.elements());
}

/// What every fused attention entry tunes over.
#[derive(Clone)]
pub(crate) struct AttentionMix {
    pub activation: Element,
    pub shape: AttentionShape,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
    /// The row classes this variant serves. Distinct ranges have distinct
    /// tuning identities and are validated on their own rows.
    pub decode_rows: Option<Range<u64>>,
}

/// `attention_decode`, for row classes up to [`DECODE_ROWS`].
pub(crate) struct AttentionDecodeTuning(pub AttentionMix);

/// `attention_prefill`, for row classes beyond [`DECODE_ROWS`].
pub(crate) struct AttentionPrefillTuning(pub AttentionMix);

/// `attention_decode_k8v4`, for row classes up to [`DECODE_ROWS`].
pub(crate) struct AttentionDecodeK8V4Tuning(pub AttentionMix);

/// `attention_prefill_k8v4`, for row classes beyond [`DECODE_ROWS`].
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
    rotary_components: Tensor,
    rotary_frequencies: Tensor,
    rotary_amplitudes: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
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
    /// Planes of `rows` history rows, viewed up to `view` rows with
    /// `appended` written.
    fn build(
        inputs: &mut TuningInputs<'_, '_>,
        mix: &AttentionMix,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, String>;
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
    fn build(
        inputs: &mut TuningInputs<'_, '_>,
        mix: &AttentionMix,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, String> {
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
    fn codes(inputs: &TuningInputs<'_, '_>, extents: &[u64], seed: u64) -> Result<Tensor, String> {
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
    ) -> Result<Tensor, String> {
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
    fn build(
        inputs: &mut TuningInputs<'_, '_>,
        mix: &AttentionMix,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, String> {
        let shape = mix.shape;
        let mut plane =
            |name: &str, build: &dyn Fn(&TuningInputs<'_, '_>) -> Result<Tensor, String>| {
                let plane = inputs.shared(
                    format!("{name}-affine-{rows}x{}x{}", shape.kv_heads, shape.width),
                    |inputs| build(inputs),
                )?;
                inputs.slab_state(plane, view, appended.clone())
            };
        let pairs = [rows, shape.kv_heads, affine_coefficients(shape.width)];
        Ok(Self {
            key_codes: plane("history_key_codes", &|inputs| {
                Self::codes(inputs, &[rows, shape.kv_heads, shape.width / 4], 0x6b)
            })?,
            key_coefficients: plane("history_key_coefficients", &|inputs| {
                Self::coefficients(inputs, &pairs, 255)
            })?,
            value_codes: plane("history_value_codes", &|inputs| {
                Self::codes(inputs, &[rows, shape.kv_heads, shape.width / 8], 0x76)
            })?,
            value_coefficients: plane("history_value_coefficients", &|inputs| {
                Self::coefficients(inputs, &pairs, 15)
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
    fn bindings(&self) -> String {
        match &self.decode_rows {
            Some(rows) => format!(
                "A={},M={}..{}",
                self.activation.name(),
                rows.start,
                rows.end
            ),
            None => format!("A={}", self.activation.name()),
        }
    }

    fn statics(&self) -> Vec<(&'static str, u64)> {
        self.shape.mix_statics().to_vec()
    }

    /// The history rows the planes of points with `context` history rows
    /// hold: the history plus the most rows any such point appends.
    fn history_rows(points: &[PointShape], context: u64) -> u64 {
        context
            + points
                .iter()
                .filter(|point| point.context == Some(context))
                .map(|point| point.rows)
                .max()
                .unwrap_or(0)
    }

    fn rotation<H: MixHistory>(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
        points: Vec<PointShape>,
    ) -> Result<Vec<AttentionMixCase<H>>, String> {
        let shape = self.shape;
        let rows = point.rows;
        let context = point
            .context
            .ok_or("attention tuning points carry a history length")?;
        // Points of one history length share planes: they all read their
        // first `context` rows, which nothing writes, and append after them.
        // A longer history has its own planes, so no point ever reads rows
        // another point appended.
        let view = context + rows;
        let history = H::build(
            inputs,
            self,
            Self::history_rows(&points, context),
            view,
            context..view,
        )?;
        let tables = inputs.batch(rows, context, 1, context)?;
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
                            .map_err(|error| error.to_string())
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
                Ok(AttentionMixCase {
                    query_norm,
                    key_norm,
                    value_norm: inputs.f32s(
                        &[shape.value_norm, width],
                        &vec![1.0; (shape.value_norm * width) as usize],
                    )?,
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
    ($tuning:ident, $module:ident, $history:ty, $serves:expr,
     |$case:ident| { $($plane:ident: $state:expr),* $(,)? }
     $(, search_starts $starts:path)?) => {
        impl $tuning {
            fn served(&self, limits: TuningLimits) -> Vec<PointShape> {
                with_contexts(
                    limits,
                    served_row_points(limits.max_rows, |rows| {
                        ($serves)(rows)
                            && self
                                .0
                                .decode_rows
                                .as_ref()
                                .is_none_or(|range| range.contains(&rows))
                    }),
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
                self.0.bindings()
            }

            fn statics(
                &self,
                _inputs: &TuningInputs<'_, '_>,
            ) -> Result<Vec<(&'static str, u64)>, String> {
                Ok(self.0.statics())
            }

            fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
                self.served(limits)
            }

            $(fn search_starts(
                &self,
                device: &Device,
                implementation: &seismic::NativeImplementation,
                statics: &seismic::NativeSpecialization,
                limits: TuningLimits,
            ) -> Vec<seismic::ParameterValues> {
                $starts(&self.0, device, implementation, statics, limits)
            })?

            fn rotation(
                &self,
                inputs: &mut TuningInputs<'_, '_>,
                point: &PointShape,
            ) -> Result<Vec<Self::Case>, String> {
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

mix_entry!(
    AttentionDecodeTuning,
    attention_decode,
    DenseMixHistory,
    |rows| rows <= DECODE_ROWS,
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
    |case| {
        history_key: case.history.key.tensor_mut(),
        history_value: case.history.value.tensor_mut(),
    }
);
mix_entry!(
    AttentionDecodeK8V4Tuning,
    attention_decode_k8v4,
    AffineMixHistory,
    |rows| rows <= DECODE_ROWS,
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
    |case| {
        history_key_codes: case.history.key_codes.tensor_mut(),
        history_key_coefficients: case.history.key_coefficients.tensor_mut(),
        history_value_codes: case.history.value_codes.tensor_mut(),
        history_value_coefficients: case.history.value_coefficients.tensor_mut(),
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_decode_has_distinct_tuning_identity_and_served_rows() {
        let mix = AttentionMix {
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
            },
            scopes: Vec::new(),
            epsilon: 1.0e-5,
            decode_rows: Some(1..2),
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
        assert_eq!(single.bindings(), "A=bf16,M=1..2");
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
        assert_eq!(verify.bindings(), "A=bf16,M=2..9");

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
            classes.extend(range);
        }
        assert_eq!(classes, (2..DECODE_ROWS + 1).collect::<Vec<_>>());
    }
}
