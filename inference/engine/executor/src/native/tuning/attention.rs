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
    cpu_projection_screening, row_points, served_row_points, with_contexts, CaseState, EntryTuning,
    PointShape, TuningInputs, TuningLimits,
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
use seismic::{Device, Element, ScreeningPoint, Tensor};
use std::ops::Range;

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

    fn screening(&self, device: &Device, points: &[PointShape]) -> Vec<ScreeningPoint> {
        cpu_projection_screening(device, points)
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

    fn screening(&self, device: &Device, points: &[PointShape]) -> Vec<ScreeningPoint> {
        cpu_projection_screening(device, points)
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
pub(crate) struct AttentionMix {
    pub activation: Element,
    pub shape: AttentionShape,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
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
/// shared by the points of one history length, ending at a point's appended
/// rows.
pub(crate) trait MixHistory: Sized {
    /// Planes of `rows` history rows shared by the points of one history
    /// length (`context`), each viewed up to `view` rows with `appended`
    /// written.
    fn build(
        inputs: &mut TuningInputs<'_, '_>,
        mix: &AttentionMix,
        context: u64,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, String>;
    fn share(&self) -> Self;
    fn states(&self) -> Vec<&CaseState>;
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
        context: u64,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, String> {
        let shape = mix.shape;
        let mut plane = |name: &str, seed: u64| {
            let plane = inputs.shared(format!("{name}-c{context}"), |inputs| {
                inputs.activation(mix.activation, &[rows, shape.kv_heads, shape.width], seed)
            })?;
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

    fn states(&self) -> Vec<&CaseState> {
        vec![&self.key, &self.value]
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
        let count = usize::try_from(extents.iter().product::<u64>())
            .map_err(|_| "tuning code plane exceeds usize")?;
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let words = (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 32) as u32
            })
            .collect::<Vec<_>>();
        inputs.u32s(extents, &words)
    }

    /// Pairs decoding codes 0..=levels onto [-1, 1]: scale 2 / levels, zero -1.
    fn coefficients(
        inputs: &TuningInputs<'_, '_>,
        extents: &[u64],
        levels: u32,
    ) -> Result<Tensor, String> {
        let count = usize::try_from(extents.iter().product::<u64>())
            .map_err(|_| "tuning coefficient plane exceeds usize")?;
        let scale = super::f16_bits(2.0 / levels as f32);
        let zero = super::f16_bits(-1.0);
        let bytes = (0..count / 2)
            .flat_map(|_| [scale.to_le_bytes(), zero.to_le_bytes()])
            .flatten()
            .collect::<Vec<_>>();
        Tensor::from_host(inputs.device, Element::f16(), extents, &bytes)
            .map_err(|error| error.to_string())
    }
}

impl MixHistory for AffineMixHistory {
    fn build(
        inputs: &mut TuningInputs<'_, '_>,
        mix: &AttentionMix,
        context: u64,
        rows: u64,
        view: u64,
        appended: Range<u64>,
    ) -> Result<Self, String> {
        let shape = mix.shape;
        let mut plane =
            |name: &str, build: &dyn Fn(&TuningInputs<'_, '_>) -> Result<Tensor, String>| {
                let plane = inputs.shared(format!("{name}-c{context}"), |inputs| build(inputs))?;
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

    fn states(&self) -> Vec<&CaseState> {
        vec![
            &self.key_codes,
            &self.key_coefficients,
            &self.value_codes,
            &self.value_coefficients,
        ]
    }
}

impl AttentionMix {
    fn bindings(&self) -> String {
        format!("A={}", self.activation.name())
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
            context,
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
     |$case:ident| { $($plane:ident: $state:expr),* $(,)? }) => {
        impl $tuning {
            fn served(limits: TuningLimits) -> Vec<PointShape> {
                with_contexts(limits, served_row_points(limits.max_rows, $serves))
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
                Self::served(limits)
            }

            fn rotation(
                &self,
                inputs: &mut TuningInputs<'_, '_>,
                point: &PointShape,
            ) -> Result<Vec<Self::Case>, String> {
                let points = Self::served(inputs.limits);
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

            fn state(case: &Self::Case) -> Vec<&CaseState> {
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
    }
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
    }
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
