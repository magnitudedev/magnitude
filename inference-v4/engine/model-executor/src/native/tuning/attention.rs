//! Tuning cases of the gated attention block: the normed Q/K/V projection,
//! the fused attention entries of each history codec (`gated_attention_decode`
//! and `gated_attention_decode_k8v4` up to [`DECODE_ROWS`] rows,
//! `gated_attention_prefill` and `gated_attention_prefill_k8v4` beyond) and the
//! output projection.
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
use crate::programs::graph::attention::{
    affine_coefficients, rotary_components, rotary_frequencies, DECODE_ROWS,
};
use crate::AttentionShape;
use magnitude_model_contracts::{MixerGeometry, RotarySemantics, WeightKind, WeightScope};
use magnitude_model_kernels::{
    attention_output, gated_attention_decode, gated_attention_decode_k8v4, gated_attention_prefill,
    gated_attention_prefill_k8v4, gated_attention_project,
};
use seismic::{Device, Element, ScreeningPoint, Tensor};
use std::ops::Range;

/// `gated_attention_project`: RMS prologue, fused query+gate | key | value
/// projection.
pub(crate) struct AttentionProjectTuning {
    pub norm: Element,
    pub query_gate: Element,
    pub key: Element,
    pub value: Element,
    pub activation: Element,
    pub shape: AttentionShape,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct AttentionProjectCase {
    hidden: Tensor,
    input_norm: Tensor,
    query_norm: Tensor,
    query_gate: Tensor,
    key: Tensor,
    value: Tensor,
    epsilon: f32,
}

impl AttentionProjectTuning {
    fn elements(&self) -> gated_attention_project::Elements {
        gated_attention_project::Elements {
            NW: self.norm,
            QW: self.query_gate,
            KW: self.key,
            VW: self.value,
            A: self.activation,
        }
    }
}

impl EntryTuning for AttentionProjectTuning {
    type Entry = gated_attention_project::Entry;
    type Case = AttentionProjectCase;

    fn bindings(&self) -> String {
        format!(
            "NW={},QW={},KW={},VW={},A={}",
            self.norm.name(),
            self.query_gate.name(),
            self.key.name(),
            self.value.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let (_, hidden) = projection_shape(inputs, &self.scopes, WeightKind::QueryGate)?;
        if hidden != shape.hidden {
            return Err(format!(
                "the query projection is {hidden} wide, the binding {}",
                shape.hidden
            ));
        }
        Ok(vec![
            ("D", shape.hidden),
            ("KV", shape.kv_heads),
            ("G", shape.group),
            ("W", shape.width),
        ])
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
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(AttentionProjectCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, self.shape.hidden],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.weight(scope, WeightKind::InputNorm)?,
                    query_norm: inputs.weight(scope, WeightKind::QueryNorm)?,
                    query_gate: inputs.weight(scope, WeightKind::QueryGate)?,
                    key: inputs.weight(scope, WeightKind::Key)?,
                    value: inputs.weight(scope, WeightKind::Value)?,
                    epsilon: self.epsilon,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> gated_attention_project::Args<'a> {
        gated_attention_project::Args {
            hidden: &case.hidden,
            input_norm: &case.input_norm,
            query_norm: &case.query_norm,
            query_gate_weight: &case.query_gate,
            key_weight: &case.key,
            value_weight: &case.value,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(gated_attention_project, this => this.elements());
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

/// `gated_attention_decode`, for row classes up to [`DECODE_ROWS`].
pub(crate) struct AttentionDecodeTuning(pub AttentionMix);

/// `gated_attention_prefill`, for row classes beyond [`DECODE_ROWS`].
pub(crate) struct AttentionPrefillTuning(pub AttentionMix);

/// `gated_attention_decode_k8v4`, for row classes up to [`DECODE_ROWS`].
pub(crate) struct AttentionDecodeK8V4Tuning(pub AttentionMix);

/// `gated_attention_prefill_k8v4`, for row classes beyond [`DECODE_ROWS`].
pub(crate) struct AttentionPrefillK8V4Tuning(pub AttentionMix);

/// One argument set of a fused entry: the inputs every codec shares, and the
/// history planes of the entry's codec.
pub(crate) struct AttentionMixCase<H> {
    query_gate: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    rotary_components: Tensor,
    rotary_frequencies: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
    history: H,
    epsilon: f32,
    scale: f32,
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
            inputs.state(
                plane
                    .slice_leading(0, view)
                    .map_err(|error| error.to_string())?,
                appended.clone(),
            )
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
                inputs.state(
                    plane
                        .slice_leading(0, view)
                        .map_err(|error| error.to_string())?,
                    appended.clone(),
                )
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

/// The model's rotary embedding: every attention block shares it.
fn rotary(inputs: &TuningInputs<'_, '_>) -> Result<RotarySemantics, String> {
    inputs
        .definition
        .geometry
        .blocks
        .iter()
        .find_map(|block| match &block.mixer {
            MixerGeometry::Attention(attention) => Some(attention.rotary.clone()),
            MixerGeometry::Recurrent(_) => None,
        })
        .ok_or_else(|| "the model has no attention block".to_owned())
}

impl AttentionMix {
    fn bindings(&self) -> String {
        format!("A={}", self.activation.name())
    }

    fn statics(&self) -> Vec<(&'static str, u64)> {
        let shape = self.shape;
        vec![
            ("KV", shape.kv_heads),
            ("G", shape.group),
            ("P", shape.rotary_pairs),
            ("S", shape.width - 2 * shape.rotary_pairs),
        ]
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
        let rotary = rotary(inputs)?;
        let components = rotary_components(&rotary)?;
        let frequencies = rotary_frequencies(&rotary);
        let pairs = components.len() as u64;
        let coordinates = tables
            .coordinates
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let visible = tables
            .visible
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let fresh = tables.fresh.iter().flatten().copied().collect::<Vec<_>>();
        let heads = shape.kv_heads * shape.group;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 3 * index as u64;
                Ok(AttentionMixCase {
                    query_gate: inputs.activation(
                        self.activation,
                        &[rows, heads * 2 * shape.width],
                        seed + 1,
                    )?,
                    key: inputs.activation(
                        self.activation,
                        &[rows, shape.kv_heads * shape.width],
                        seed + 2,
                    )?,
                    value: inputs.activation(
                        self.activation,
                        &[rows, shape.kv_heads * shape.width],
                        seed + 3,
                    )?,
                    query_norm: inputs.weight(scope, WeightKind::QueryNorm)?,
                    key_norm: inputs.weight(scope, WeightKind::KeyNorm)?,
                    rotary_components: inputs.i32s(&[pairs], &components)?,
                    rotary_frequencies: inputs.f32s(&[pairs], &frequencies)?,
                    coordinates: inputs.i32s(&[rows, 4], &coordinates)?,
                    visible: inputs.i32s(&[rows, segments, 2], &visible)?,
                    fresh: inputs.i32s(&[rows, 2], &fresh)?,
                    destinations: inputs.i32s(&[rows], &tables.destinations)?,
                    history: history.share(),
                    epsilon: self.epsilon,
                    scale: 1.0 / (shape.width as f32).sqrt(),
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
                    query_gate: &$case.query_gate,
                    key: &$case.key,
                    value: &$case.value,
                    query_norm: &$case.query_norm,
                    key_norm: &$case.key_norm,
                    rotary_components: &$case.rotary_components,
                    rotary_frequencies: &$case.rotary_frequencies,
                    coordinates: &$case.coordinates,
                    visible: &$case.visible,
                    fresh: &$case.fresh,
                    destinations: &$case.destinations,
                    $($plane: $state,)*
                    epsilon: $case.epsilon,
                    scale: $case.scale,
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
    gated_attention_decode,
    DenseMixHistory,
    |rows| rows <= DECODE_ROWS,
    |case| {
        history_key: case.history.key.tensor_mut(),
        history_value: case.history.value.tensor_mut(),
    }
);
mix_entry!(
    AttentionPrefillTuning,
    gated_attention_prefill,
    DenseMixHistory,
    |rows| rows > DECODE_ROWS,
    |case| {
        history_key: case.history.key.tensor_mut(),
        history_value: case.history.value.tensor_mut(),
    }
);
mix_entry!(
    AttentionDecodeK8V4Tuning,
    gated_attention_decode_k8v4,
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
    gated_attention_prefill_k8v4,
    AffineMixHistory,
    |rows| rows > DECODE_ROWS,
    |case| {
        history_key_codes: case.history.key_codes.tensor_mut(),
        history_key_coefficients: case.history.key_coefficients.tensor_mut(),
        history_value_codes: case.history.value_codes.tensor_mut(),
        history_value_coefficients: case.history.value_coefficients.tensor_mut(),
    }
);
